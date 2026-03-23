"""
Download Open Interest data from B3 (Brazilian Stock Exchange).
Multiple sources with fallback.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO, BytesIO
import zipfile
import os

# Directory to cache downloaded COTAHIST files (avoids re-downloading)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".b3_cache")


def get_previous_business_day() -> str:
    """Get previous business day in YYYY-MM-DD format."""
    d = datetime.now()
    # Go back to previous day
    d -= timedelta(days=1)
    # Skip weekends
    while d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        d -= timedelta(days=1)
    return d.strftime('%Y-%m-%d')


# ============================================================
# Source 1: B3 Daily Trading Data (rapinegocios)
# ============================================================
def fetch_b3_trading_data(date: str = None) -> pd.DataFrame:
    """
    Fetch daily trading data from B3 API.
    Contains all traded instruments including options.
    """
    if date is None:
        date = get_previous_business_day()

    url = f"https://arquivos.b3.com.br/rapinegocios/tickercsv/{date}"

    print(f"[>] [Source 1] Fetching B3 trading data for {date}...")
    print(f"   URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text),
            sep=';',
            encoding='latin-1',
            on_bad_lines='skip',
        )

        print(f"   [OK] Loaded {len(df)} records")
        print(f"   Columns: {list(df.columns)}")
        return df

    except requests.exceptions.HTTPError as e:
        print(f"   [X] HTTP Error: {e}")
        print(f"   Try a different date. B3 may not have data for {date}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"   [X] Error: {e}")
        return pd.DataFrame()


# ============================================================
# Source 2: B3 Historical Series (Série Histórica)
# ============================================================
def fetch_b3_historical_file(date: str = None) -> pd.DataFrame:
    """
    Fetch from B3 historical series.
    File format: COTAHIST_DDDMMYYYY.ZIP containing COTAHIST_DDDMMYYYY.TXT

    The TXT file is fixed-width format with OI data.
    Uses local cache to avoid re-downloading the same date.
    """
    if date is None:
        date = get_previous_business_day()

    d = datetime.strptime(date, '%Y-%m-%d')
    date_str = d.strftime('%d%m%Y')

    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{date_str}.ZIP"
    zip_filename = f"COTAHIST_D{date_str}.ZIP"

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached_zip = os.path.join(CACHE_DIR, zip_filename)

    print(f"\n[>] [Source 2] B3 historical file for {date}...")

    try:
        # --- Check if file already exists in cache ---
        if os.path.exists(cached_zip) and os.path.getsize(cached_zip) > 0:
            print(f"   [CACHE HIT] Using local file: {cached_zip}")
            with zipfile.ZipFile(cached_zip) as z:
                txt_files = [f for f in z.namelist() if f.endswith('.TXT')]
                if not txt_files:
                    print("   [X] No TXT file in cached ZIP — deleting and retrying")
                    os.remove(cached_zip)
                else:
                    with z.open(txt_files[0]) as txt_file:
                        content = txt_file.read().decode('latin-1')
                    return _parse_cotahist_content(content)

        # --- Download from B3 ---
        print(f"   [DOWNLOAD] {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Save to cache
        with open(cached_zip, 'wb') as f:
            f.write(response.content)
        print(f"   [SAVED] {cached_zip} ({len(response.content) / 1024:.0f} KB)")

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            txt_files = [f for f in z.namelist() if f.endswith('.TXT')]
            if not txt_files:
                print("   [X] No TXT file in ZIP")
                return pd.DataFrame()

            with z.open(txt_files[0]) as txt_file:
                content = txt_file.read().decode('latin-1')

        return _parse_cotahist_content(content)

    except requests.exceptions.HTTPError:
        print(f"   [X] File not found. Try a different date.")
        return pd.DataFrame()
    except Exception as e:
        print(f"   [X] Error: {e}")
        return pd.DataFrame()


def _parse_cotahist_content(content: str) -> pd.DataFrame:
    """Parse COTAHIST fixed-width TXT content into a DataFrame of options."""

    lines = content.split('\n')
    records = []

    for line in lines:
        if len(line) < 170:
            continue
        if line[0:2] != '01':  # only data records
            continue

        record = {
            'date': line[2:10].strip(),
            'bdi_code': line[10:12].strip(),
            'ticker': line[12:24].strip(),
            'market_type': line[24:27].strip(),
            'company': line[27:39].strip(),
            'currency': line[52:56].strip(),
            'open': int(line[56:69]) / 100.0,
            'high': int(line[69:82]) / 100.0,
            'low': int(line[82:95]) / 100.0,
            'avg': int(line[95:108]) / 100.0,
            'close': int(line[108:121]) / 100.0,
            'best_bid': int(line[121:134]) / 100.0,
            'best_ask': int(line[134:147]) / 100.0,
            'num_trades': int(line[147:152]),
            'quantity': int(line[152:170]),       # ← This is OI for options
            'volume': int(line[170:188]) / 100.0,
            'strike': int(line[188:201]) / 100.0,
            'expiration': line[202:210].strip(),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Market type 70 = Options on Stocks (Calls)
    # Market type 80 = Options on Stocks (Puts)
    options = df[df['market_type'].isin(['070', '70', '080', '80'])].copy()

    print(f"   [OK] Loaded {len(df)} total records, {len(options)} options")
    return options


# ============================================================
# Source 3: B3 Open Data Portal
# ============================================================
def fetch_b3_open_data(date: str = None) -> pd.DataFrame:
    """
    Fetch from B3 Open Data portal.
    https://dados.b3.com.br/
    """
    if date is None:
        date = get_previous_business_day()

    # B3 dados abertos API
    url = f"https://arquivos.b3.com.br/apinegocios/tickercsv/{date}"

    print(f"\n[>] [Source 3] Fetching B3 open data...")
    print(f"   URL: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(
            StringIO(response.text),
            sep=';',
            encoding='latin-1',
            on_bad_lines='skip',
        )

        print(f"   [OK] Loaded {len(df)} records")
        return df

    except Exception as e:
        print(f"   [X] Error: {e}")
        return pd.DataFrame()


# ============================================================
# Extract Options OI for a Specific Underlying
# ============================================================
def extract_options_oi(
    df: pd.DataFrame,
    underlying: str = "BOVA11",
) -> pd.DataFrame:
    """
    From raw B3 data, extract options OI for a given underlying.

    B3 Option Ticker Format:
        PETRX999
        ||||└─── Strike price code
        |||└──── Series letter (A-L = Calls, M-X = Puts)
        └└└───── First 4 chars of underlying

    Series Letters:
        Calls: A=Jan, B=Feb, C=Mar, D=Apr, E=May, F=Jun,
               G=Jul, H=Aug, I=Sep, J=Oct, K=Nov, L=Dec
        Puts:  M=Jan, N=Feb, O=Mar, P=Apr, Q=May, R=Jun,
               S=Jul, T=Aug, U=Sep, V=Oct, W=Nov, X=Dec
    """
    prefix = underlying[:4].upper()

    call_letters = 'ABCDEFGHIJKL'
    put_letters = 'MNOPQRSTUVWX'

    # Find the ticker column (varies by source)
    ticker_col = None
    for col in df.columns:
        if 'ticker' in col.lower() or 'codneg' in col.lower() or 'codigo' in col.lower():
            ticker_col = col
            break

    if ticker_col is None:
        print(f"   Available columns: {list(df.columns)}")
        print(f"   [X] Could not find ticker column")
        return pd.DataFrame()

    # Filter options for this underlying
    mask = df[ticker_col].str.startswith(prefix, na=False)
    options = df[mask].copy()

    if options.empty:
        print(f"   [X] No options found for {underlying}")
        return pd.DataFrame()

    # Parse each option ticker
    results = []
    for _, row in options.iterrows():
        ticker = str(row[ticker_col]).strip()
        if len(ticker) < 6:
            continue

        series_letter = ticker[4].upper()

        # Determine call/put
        if series_letter in call_letters:
            opt_type = 'call'
            month = call_letters.index(series_letter) + 1
        elif series_letter in put_letters:
            opt_type = 'put'
            month = put_letters.index(series_letter) + 1
        else:
            continue

        # OI column — try multiple possible names
        oi = 0
        for oi_col in ['quantity', 'QUATOT', 'QuantidadeAberta',
                        'ContratoAberto', 'OpenInterest', 'OI']:
            if oi_col in row.index:
                try:
                    oi = float(row[oi_col])
                except (ValueError, TypeError):
                    pass
                break

        # Strike — try multiple possible names
        strike = 0
        for strike_col in ['strike', 'PrecoExercicio', 'Strike',
                           'PREEXE', 'PrecoExerc']:
            if strike_col in row.index:
                try:
                    strike = float(row[strike_col])
                except (ValueError, TypeError):
                    pass
                break

        # Volume
        volume = 0
        for vol_col in ['volume', 'Volume', 'VOLTOT', 'VolumeTotalNegociado']:
            if vol_col in row.index:
                try:
                    volume = float(row[vol_col])
                except (ValueError, TypeError):
                    pass
                break

        # Close price
        close = 0
        for close_col in ['close', 'Close', 'PrecoUltimo', 'PREULT']:
            if close_col in row.index:
                try:
                    close = float(row[close_col])
                except (ValueError, TypeError):
                    pass
                break

        results.append({
            'ticker': ticker,
            'type': opt_type,
            'series': series_letter,
            'month': month,
            'strike': strike,
            'oi': oi,
            'volume': volume,
            'close': close,
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        # Summary
        calls = result_df[result_df['type'] == 'call']
        puts = result_df[result_df['type'] == 'put']
        print(f"\n   [*] Options for {underlying}:")
        print(f"      Calls: {len(calls)} strikes | Total OI: {calls['oi'].sum():,.0f}")
        print(f"      Puts:  {len(puts)} strikes | Total OI: {puts['oi'].sum():,.0f}")

    return result_df


# ============================================================
# Build Chain Ready for GEX Calculation
# ============================================================
def build_gex_chain_from_b3(
    underlying: str = "BOVA11",
    date: str = None,
) -> pd.DataFrame:
    """
    Complete pipeline: B3 → Options chain → Ready for GEX.
    Tries multiple B3 data sources with fallback.
    """
    print("=" * 55)
    print(f"  B3 Open Interest Downloader — {underlying}")
    print("=" * 55)

    # Try sources in order
    df = pd.DataFrame()

    # Source 1: rapinegocios
    df = fetch_b3_trading_data(date)

    # Source 2: Historical file (fallback)
    if df.empty:
        df = fetch_b3_historical_file(date)

    # Source 3: Open data (fallback)
    if df.empty:
        df = fetch_b3_open_data(date)

    if df.empty:
        print("\n[X] Could not fetch data from any B3 source.")
        print("   Try manually downloading from:")
        print("   https://www.b3.com.br/pt_br/market-data-e-indices/")
        return pd.DataFrame()

    # Extract options OI
    options = extract_options_oi(df, underlying)

    if options.empty:
        return pd.DataFrame()

    # Pivot to GEX-ready format
    calls = options[options['type'] == 'call'][['strike', 'oi', 'volume']].rename(
        columns={'oi': 'call_oi', 'volume': 'call_vol'}
    )
    puts = options[options['type'] == 'put'][['strike', 'oi', 'volume']].rename(
        columns={'oi': 'put_oi', 'volume': 'put_vol'}
    )

    chain = pd.merge(calls, puts, on='strike', how='outer').fillna(0)
    chain = chain.sort_values('strike').reset_index(drop=True)

    print(f"\n[*] Merged chain: {len(chain)} strikes")
    print(chain.head())

    # Add IV placeholder (needs BS inversion with real prices)
    chain['call_iv'] = 0.35
    chain['put_iv'] = 0.35

    # Add expiration
    now = datetime.now()
    chain['expiration_date'] = datetime(now.year, now.month + 1, 17)

    print(f"\n[OK] GEX-ready chain: {len(chain)} strikes")
    print(chain[['strike', 'call_oi', 'put_oi','expiration_date']].head().to_string(index=False))

    return chain


# ============================================================
# Source 4: opcoes.net.br — Free Real-Time Options Data
# ============================================================
def fetch_opcoes_net_br(underlying: str = "BOVA11") -> pd.DataFrame:
    """
    Fetch live options chain from opcoes.net.br (free Brazilian options aggregator).
    Returns DataFrame with columns: ticker, type, strike, last, num_trades,
    volume, iv, delta, gamma, theta, vega, expiration.

    NOTE: Only returns data during B3 market hours (~10:00-17:00 BRT).
          Outside market hours, returns empty DataFrame.
    """
    prefix = underlying[:4].upper() if underlying else "BOVA"

    url = f"https://opcoes.net.br/listaopcoes/completa?idAcao={underlying}&liession=1&cotession=1"

    print(f"\n[>] [Source 4] Fetching from opcoes.net.br for {underlying}...")

    try:
        s = requests.Session()
        s.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json',
        })
        r = s.get(url, timeout=15)
        r.raise_for_status()

        data = r.json()
        if not data.get('success'):
            print("   [X] API returned success=false")
            return pd.DataFrame()

        options_list = data.get('data', {}).get('cotacoesOpcoes', [])
        columns_list = data.get('data', {}).get('columns', [])

        if not options_list:
            print("   [X] No options data (market may be closed)")
            return pd.DataFrame()

        col_names = [c.get('name', f'col{i}') for i, c in enumerate(columns_list)]
        print(f"   [OK] {len(options_list)} options found")
        print(f"   Columns: {col_names}")

        # Build DataFrame from the list responses
        # Each option is a list matching the columns order
        records = []
        for opt in options_list:
            if isinstance(opt, list) and len(opt) >= len(col_names):
                record = dict(zip(col_names, opt))
                records.append(record)
            elif isinstance(opt, dict):
                records.append(opt)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Standardize column names for downstream use
        rename_map = {
            'ticker': 'ticker',
            'tipo': 'Tipo',
            'strike': 'Strike',
            'ultimo': 'last',
            'numerodenegocios': 'num_trades',
            'volumenegociado': 'volume',
            'vol.implicita': 'iv',
            'delta': 'delta',
            'gamma': 'gamma',
            'theta': 'theta',
            'vega': 'vega',
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Convert numeric columns from Brazilian format (comma decimal)
        for col in ['Strike', 'last', 'iv', 'delta', 'gamma', 'theta', 'vega', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
                    errors='coerce'
                )

        return df

    except requests.exceptions.HTTPError as e:
        print(f"   [X] HTTP Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"   [X] Error: {e}")
        return pd.DataFrame()


# ============================================================
# Multi-day COTAHIST Volume Accumulation (OI Proxy)
# ============================================================
def fetch_multiday_volume(
    underlying: str = "BOVA11",
    num_days: int = 5,
) -> pd.DataFrame:
    """
    Download COTAHIST for the last N business days and accumulate option volumes
    per ticker. The accumulated volume is a rough proxy for Open Interest because
    options with high OI tend to have consistently high daily volume.

    Returns DataFrame with columns: ticker, accumulated_volume, avg_daily_volume,
        days_active, strike, expiration, market_type
    """
    prefix = underlying[:4].upper()
    call_letters = set('ABCDEFGHIJKL')
    put_letters = set('MNOPQRSTUVWX')

    print(f"\n[>] Fetching {num_days}-day volume accumulation for {underlying}...")

    all_records = []
    dates_fetched = 0
    d = datetime.now() - timedelta(days=1)

    while dates_fetched < num_days:
        # Skip weekends
        while d.weekday() >= 5:
            d -= timedelta(days=1)

        date_str = d.strftime('%Y-%m-%d')
        try:
            day_data = fetch_b3_historical_file(date_str)
            if not day_data.empty:
                # Filter for this underlying's options
                opts = day_data[day_data['ticker'].str.startswith(prefix, na=False)].copy()
                if not opts.empty:
                    opts['fetch_date'] = date_str
                    all_records.append(opts)
                    dates_fetched += 1
                    print(f"   Day {dates_fetched}/{num_days}: {date_str} -> {len(opts)} options")
        except Exception as e:
            print(f"   [X] {date_str}: {e}")

        d -= timedelta(days=1)

        # Safety: don't go back more than 15 calendar days
        if (datetime.now() - d).days > 15:
            break

    if not all_records:
        print("   [X] No data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)

    # Aggregate by ticker
    agg = combined.groupby('ticker').agg(
        accumulated_volume=('quantity', 'sum'),
        avg_daily_volume=('quantity', 'mean'),
        days_active=('fetch_date', 'nunique'),
        strike=('strike', 'first'),
        expiration=('expiration', 'first'),
        market_type=('market_type', 'first'),
    ).reset_index()

    # Add call/put classification
    def classify(ticker):
        if len(ticker) > 4:
            letter = ticker[4].upper()
            if letter in call_letters:
                return 'CALL'
            elif letter in put_letters:
                return 'PUT'
        return None

    agg['type'] = agg['ticker'].apply(classify)
    agg = agg.dropna(subset=['type'])

    print(f"   [OK] Accumulated {dates_fetched} days: {len(agg)} unique options")
    print(f"   Acc. volume range: {agg['accumulated_volume'].min():.0f} - {agg['accumulated_volume'].max():.0f}")

    return agg


# ============================================================
# External OI CSV Loader
# ============================================================
def load_external_oi(filepath: str) -> pd.DataFrame:
    """
    Load Open Interest data from an external CSV file.
    The user can generate this from any source (B3 website, Bloomberg,
    paid data feeds, etc.).

    Expected CSV columns (at minimum):
        ticker  - Option ticker (e.g., BOVAC1)
        oi      - Open Interest (number of contracts)

    Optional columns:
        strike  - Strike price
        type    - CALL or PUT
        expiration - Expiry date
    """
    if not os.path.exists(filepath):
        print(f"   [X] OI file not found: {filepath}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        print(f"   [OK] Loaded external OI: {len(df)} records from {filepath}")

        # Standardize column names (case-insensitive)
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ('ticker', 'symbol', 'codigo', 'codneg'):
                col_map[c] = 'ticker'
            elif cl in ('oi', 'openinterest', 'open_interest', 'contratos_abertos', 'posicao'):
                col_map[c] = 'oi'
            elif cl in ('strike', 'exercicio', 'preco_exercicio'):
                col_map[c] = 'strike'
            elif cl in ('type', 'tipo', 'call_put'):
                col_map[c] = 'type'
            elif cl in ('expiration', 'vencimento', 'expiry'):
                col_map[c] = 'expiration'

        df = df.rename(columns=col_map)

        if 'ticker' not in df.columns or 'oi' not in df.columns:
            print(f"   [X] CSV must have 'ticker' and 'oi' columns. Found: {list(df.columns)}")
            return pd.DataFrame()

        df['oi'] = pd.to_numeric(df['oi'], errors='coerce').fillna(0)
        print(f"   OI range: {df['oi'].min():.0f} - {df['oi'].max():.0f}")
        return df

    except Exception as e:
        print(f"   [X] Error loading OI file: {e}")
        return pd.DataFrame()


# ============================================================
# Unified OI Fetcher — tries all sources
# ============================================================
def fetch_open_interest(
    underlying: str = "BOVA11",
    oi_csv_path: str = None,
    multiday_days: int = 5,
) -> pd.DataFrame:
    """
    Get the best available OI data. Priority:
      1. External CSV file (user-provided real OI)
      2. Multi-day COTAHIST volume accumulation (OI proxy)

    Returns DataFrame with columns: ticker, oi
    """
    # Priority 1: External OI file
    if oi_csv_path:
        oi = load_external_oi(oi_csv_path)
        if not oi.empty:
            oi['oi_source'] = 'external_csv'
            return oi[['ticker', 'oi', 'oi_source']]

    # Priority 2: Multi-day volume accumulation
    multiday = fetch_multiday_volume(underlying, num_days=multiday_days)
    if not multiday.empty:
        result = multiday[['ticker', 'accumulated_volume']].copy()
        result = result.rename(columns={'accumulated_volume': 'oi'})
        result['oi_source'] = f'multiday_volume_{multiday_days}d'
        return result

    print("   [X] No OI data available from any source")
    return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    UNDERLYING = "BOVA11"
    chain = build_gex_chain_from_b3(UNDERLYING)

    if not chain.empty:
        # Save to CSV for inspection
        chain.to_csv(f"{UNDERLYING}_options_chain.csv", index=False)
        print(f"\nSaved to: {UNDERLYING}_options_chain.csv")
        print(f"\n   Next step: Pass this to calculate_gex_by_strike(chain, spot_price)")