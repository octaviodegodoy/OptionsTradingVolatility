# -*- coding: utf-8 -*-
"""
BOVA11 Options Analytics (B3 format) — Outspoken Market version
---------------------------------------------------------------
Reads Excel files from B3 with columns like:
['Ticker', 'Vencimento', 'Tipo', 'Strike', 'Último', 'Vol. Impl. (%)',
 'Delta', 'Gamma', 'Theta ($)', 'Vega', 'Tit.', 'Lan.', 'Vol. Financeiro']

Performs:
- Global and range-based Put/Call Ratio
- IV skew (OTM puts vs OTM calls)
- Notional by strike (volume financeiro)
- Gamma Exposure (Customer/Dealer)
- Call/Put walls and Gamma Flip
"""
import os
import numpy as np
import pandas as pd
import asyncio
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mt5_connector import MT5Connector

# Resolve paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNDERLYING = "PETR4"

# ------------------------------------------------------------
# Função principal de análise
# ------------------------------------------------------------
# Esta função carrega e analisa dados de opções para PETR4 (ou arquivos semelhantes no formato B3).
# O parâmetro 'spot' é passado para alinhar a análise com o preço atual do ativo subjacente.
# O objetivo é realizar várias análises como razão Put/Call, skew de volatilidade implícita, exposição gamma, etc., para avaliar o sentimento de mercado e posicionamento.
async def analyze_options(file_path: str, spot: float):
       """
       Load and analyze options data for PETR4 (or similar B3 format files).
       Spot is passed as a parameter so the analysis aligns with current price.
       """
      
       # Lê o arquivo Excel no formato B3 usando o engine openpyxl para garantir compatibilidade.
       # O objetivo é carregar os dados brutos das opções em um DataFrame do pandas.
       df = pd.read_excel(file_path, engine="openpyxl")
       #df = pd.read_excel(file_path, engine="openpyxl", header=1)  # Linha comentada: alternativa para ler com cabeçalho na segunda linha, se necessário.
   
       # Normaliza os nomes das colunas removendo espaços extras e substituindo múltiplos espaços por um único.
       # O objetivo é padronizar os nomes das colunas para evitar erros em acessos subsequentes.
       df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
       
       # Renomeia colunas comuns para rótulos padronizados, facilitando o uso no código.
       # O objetivo é tornar o código mais legível e consistente, independentemente de variações nos nomes originais.
       df.rename(columns={
           'Vol. Impl. (%)': 'IV',
           'Último': 'Ultimo',
           'Vol. Financeiro': 'VolFin',
           'Strike': 'Strike'
       }, inplace=True)
   
       # Mantém apenas as colunas relevantes para a análise, descartando as demais.
       # O objetivo é reduzir o DataFrame ao essencial, otimizando o processamento.
       df = df[['Ticker','Tipo','Strike','Ultimo','IV','Delta','Gamma',
                'Theta ($)','Vega','Tit.','Lanç.','VolFin']].copy()
   
       # --- Limpeza de dados ---
       # Converte os valores para numéricos, tratando erros como NaN para evitar falhas.
       # O objetivo é garantir que as colunas numéricas estejam no formato correto para cálculos matemáticos.
       df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce')
       df['IV'] = pd.to_numeric(df['IV'], errors='coerce') / 100.0  # Converte porcentagem para decimal, facilitando cálculos de volatilidade.
       df['Gamma'] = pd.to_numeric(df['Gamma'], errors='coerce')
       df['VolFin'] = pd.to_numeric(df['VolFin'], errors='coerce')
       df['Tit.'] = pd.to_numeric(df['Tit.'], errors='coerce')      # Converte o número de contratos mantidos (titulares).
       df['Lanç.'] = pd.to_numeric(df['Lanç.'], errors='coerce')    # Converte o número de contratos lançados (vendidos).
       # Remove linhas inválidas onde Strike, IV ou Gamma são NaN.
       # O objetivo é limpar o dataset, removendo dados incompletos que poderiam distorcer as análises.
       df = df.dropna(subset=['Strike', 'IV', 'Gamma'])             
   
       # Divide o DataFrame entre opções de call e put com base na coluna 'Tipo'.
       # O objetivo é separar as opções para análises específicas por tipo.
       calls = df[df['Tipo'].str.upper().str.contains('CALL')]
       puts  = df[df['Tipo'].str.upper().str.contains('PUT')]
   
       # ------------------------------------------------------------
       # RAZÃO PUT/CALL GLOBAL (medidor de sentimento de mercado)
       # ------------------------------------------------------------
       # Calcula o total de contratos de calls.
       # O objetivo é somar o interesse aberto em calls para comparação com puts.
       total_calls = calls['Tit.'].sum()
       # Calcula o total de contratos de puts.
       total_puts  = puts['Tit.'].sum()
       # Calcula a razão Put/Call global, tratando divisão por zero.
       # O objetivo é obter uma métrica de sentimento: alto PCR indica medo (bearish), baixo indica otimismo (bullish).
       pcr_global = total_puts / total_calls if total_calls > 0 else np.nan
   
       # Imprime o cabeçalho da seção de PCR global.
       print(f"\n===== STOCK OPTIONS — Global PCR =====")
       # Imprime o preço spot atual.
       print(f"Spot: {spot:.2f}")
       # Imprime o total de calls formatado.
       print(f"Total Calls: {total_calls:,.2f}")
       # Imprime o total de puts formatado.
       print(f"Total Puts : {total_puts:,.2f}")
       # Imprime a razão Put/Call formatada.
       print(f"Put/Call Ratio: {pcr_global:.2f}")
   
       # ------------------------------------------------------------
       # SKEW DE VOLATILIDADE IMPLÍCITA — OTM puts vs OTM calls
       # ------------------------------------------------------------
       # Filtra puts out-of-the-money (strikes abaixo do spot).
       # O objetivo é isolar puts OTM para calcular IV média.
       puts_otm  = puts[puts['Strike'] < spot]   
       # Filtra calls out-of-the-money (strikes acima do spot).
       calls_otm = calls[calls['Strike'] > spot] 
       # Calcula a IV média de puts OTM em porcentagem.
       iv_puts_otm  = puts_otm['IV'].mean() * 100
       # Calcula a IV média de calls OTM em porcentagem.
       iv_calls_otm = calls_otm['IV'].mean() * 100
       # Calcula o skew: positivo indica puts mais caros (medo).
       # O objetivo é medir o viés de volatilidade, indicando hedging ou especulação.
       iv_skew = iv_puts_otm - iv_calls_otm      
   
       # Imprime o cabeçalho da seção de skew de IV.
       print(f"\n===== Implied Volatility Skew =====")
       # Imprime IV de puts OTM.
       print(f"OTM Puts IV : {iv_puts_otm:.2f}%")
       # Imprime IV de calls OTM.
       print(f"OTM Calls IV: {iv_calls_otm:.2f}%")
       # Imprime o skew calculado.
       print(f"Skew (Puts - Calls): {iv_skew:.2f}%")
   
       # ------------------------------------------------------------
       # RAZÃO PUT/CALL por faixas de strike (bins ao redor do spot)
       # ------------------------------------------------------------
       # Define as faixas de strike relativas ao spot.
       # O objetivo é categorizar strikes em regiões como deep OTM, near OTM, ATM, etc., para análise segmentada.
       bins = [
           (0, 0.95*spot),          # Deep OTM puts
           (0.95*spot, 0.99*spot),  # Near OTM puts
           (0.99*spot, 1.01*spot),  # ATM range
           (1.01*spot, 1.05*spot),  # Near OTM calls
           (1.05*spot, np.inf),     # Far OTM calls
       ]
       # Inicializa uma lista para armazenar os resultados por faixa.
       rows = []
       # Itera sobre cada faixa de bins.
       for (low, high) in bins:
           # Cria um rótulo para a faixa.
           label = f"{low:.2f}-{high if np.isfinite(high) else '∞'}"
           # Soma contratos de calls na faixa.
           c = calls[(calls['Strike']>=low)&(calls['Strike']<high)]['Tit.'].sum()
           # Soma contratos de puts na faixa.
           p = puts[(puts['Strike']>=low)&(puts['Strike']<high)]['Tit.'].sum()
           # Calcula PCR para a faixa, tratando divisão por zero.
           pcr = p/c if c>0 else np.nan
           # Adiciona a linha à lista.
           rows.append((label, c, p, pcr))
       # Cria um DataFrame com os resultados de PCR por faixa.
       # O objetivo é visualizar o sentimento por regiões de strike.
       df_pcr = pd.DataFrame(rows, columns=['Strike Range','Calls','Puts','PCR'])
       # Imprime o cabeçalho da seção.
       print(f"\n===== PCR by Strike Range =====")
       # Imprime o DataFrame.
       print(df_pcr)
   
       # ------------------------------------------------------------
       # NOTIONAL (volume financeiro por strike)
       # ------------------------------------------------------------
       # Agrupa o volume financeiro por strike e tipo, desempilhando para colunas.
       # O objetivo é calcular o notional por strike para visualização.
       vol_by_strike = df.groupby(['Strike','Tipo'])['VolFin'].sum().unstack(fill_value=0)
       # Plota um gráfico de barras empilhadas para o volume por strike.
       # O objetivo é visualizar o volume financeiro de calls e puts por strike.
       vol_by_strike.plot(kind='bar', stacked=True, figsize=(12,12),
                          color=['#2563EB','#EF4444'], alpha=0.7)
       # Adiciona uma linha vertical no strike mais próximo do spot como âncora visual.
       plt.axvline(np.argmin(np.abs(vol_by_strike.index - spot)), color='black', linestyle='--')
       # Define o título do gráfico.
       plt.title("Volume Financeiro por Strike — Ativo")
       # Define o rótulo do eixo Y.
       plt.ylabel("Volume (R$)")
       # Define o rótulo do eixo X.
       plt.xlabel("Strike")
       # Ajusta o layout para evitar cortes.
       plt.tight_layout()
       # Exibe o gráfico.
       plt.show()
   
       # ------------------------------------------------------------
       # EXPOSIÇÃO GAMMA (Cliente vs Dealer)
       # ------------------------------------------------------------
       # Calcula a exposição gamma do cliente: gamma * (spot^2) * contratos.
       df['GEX_customer'] = df['Gamma'] * (spot**2) * df['Tit.']
       # Ajusta o sinal: positivo para calls, negativo para puts.
       # O objetivo é refletir o impacto direcional da gamma.
       df['GEX_customer'] = df['GEX_customer'] * np.where(df['Tipo'].str.upper().str.contains('CALL'), 1, -1)
       # Calcula a exposição gamma do dealer como oposta à do cliente - isto é uma convenção do mercado
       df['GEX_dealer']   = -df['GEX_customer']   
   
       # Agrega a GEX por strike para cliente e dealer.
       # O objetivo é obter totais por strike para análise e plotagem.
       gex_by_strike = df.groupby('Strike', as_index=False).agg(
           GEX_customer=('GEX_customer','sum'),
           GEX_dealer=('GEX_dealer','sum')
       ).sort_values('Strike')
   
       # ------------------------------------------------------------
       # PAREDES DE CALL/PUT — strike com máximo interesse aberto por lado
       # ------------------------------------------------------------
       # Encontra o strike com máximo Tit. para calls (parede de call).
       # O objetivo é identificar níveis de resistência/suporte baseados em OI.
       call_wall = calls.groupby('Strike')['Tit.'].sum().idxmax()
       # Encontra o strike com máximo Tit. para puts (parede de put).
       put_wall  = puts.groupby('Strike')['Tit.'].sum().idxmax()
   
       # ------------------------------------------------------------
       # GAMMA FLIP — cruzamento aproximado de zero na GEX do cliente
       # ------------------------------------------------------------
       # Extrai valores de GEX e strikes.
       gvals = gex_by_strike['GEX_customer'].to_numpy()
       strikes = gex_by_strike['Strike'].to_numpy()
       # Verifica se há dados suficientes para suavização.
       if len(strikes) > 3:
           # Aplica média móvel de 3 pontos para suavizar a curva.
           smooth = pd.Series(gvals).rolling(3, center=True, min_periods=1).mean()
           # Encontra o strike mais próximo de zero na curva suavizada.
           # O objetivo é identificar o ponto de inversão de gamma (flip), onde o comportamento do mercado muda.
           gamma_flip = strikes[np.argmin(np.abs(smooth))]  
       else:
           # Define como NaN se não houver dados suficientes.
           gamma_flip = np.nan
   
       # Imprime o cabeçalho da seção de paredes.
       print(f"\n===== Call/Put Walls =====")
       # Imprime a parede de call.
       print(f"Call Wall: {call_wall:.2f}")
       # Imprime a parede de put.
       print(f"Put  Wall: {put_wall:.2f}")
       # Imprime o gamma flip aproximado.
       print(f"Gamma Flip (approx): {gamma_flip:.2f}")
   
       # ------------------------------------------------------------
       # GRÁFICO DE EXPOSIÇÃO GAMMA (mapa visual de posicionamento)
       # ------------------------------------------------------------
       # Extrai strikes e valores de GEX em milhões.
       strikes = gex_by_strike['Strike'].to_numpy(dtype=float)
       gvals = (gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)
   
       # Calcula largura dinâmica das barras proporcional ao espaçamento de strikes.
       # O objetivo é evitar sobreposição em gráficos com strikes irregulares.
       u = np.unique(strikes)
       if len(u) >= 3:
           step = np.median(np.diff(u))
       elif len(u) == 2:
           step = abs(u[1] - u[0])
       else:
           step = 0.1
       bar_width = step * 0.6  
   
       # Aplica suavização de 3 pontos na GEX.
       smooth = pd.Series(gvals).rolling(3, center=True, min_periods=1).mean().values
   
       # Cria uma figura e eixo para o gráfico.
       fig, ax = plt.subplots(figsize=(10, 10))
       # Coloca o grid abaixo das barras.
       ax.set_axisbelow(True)
   
       # Define cores das barras: verde para gamma positiva, vermelho para negativa.
       bar_colors = np.where(gvals >= 0, "#10B981", "#EF4444")
       # Plota as barras de GEX por strike.
       # O objetivo é visualizar a distribuição de gamma.
       ax.bar(strikes, gvals, width=bar_width, align="center",
              color=bar_colors, edgecolor="none", alpha=0.55, zorder=3,
              label="Gamma Exposure by Strike")
   
       # Plota a linha suavizada para interpretação visual mais fácil.
       ax.plot(strikes, smooth, color="#2563EB", lw=2.2, zorder=4,
               label="Aggregate Gamma Exposure (smoothed)")
   
       # Adiciona marcadores verticais: spot, gamma flip, paredes.
       # O objetivo é destacar níveis chave no gráfico.
       ax.axvline(spot, color="green", lw=1.2, zorder=5, label="Spot")
       if np.isfinite(gamma_flip):
           ax.axvline(gamma_flip, color="#DC2626", lw=1.2, zorder=5,
                      label=f"Gamma Flip (approx): {gamma_flip:.2f}")
   
       # Adiciona sombreamento para regimes de gamma positiva vs negativa.
       if len(strikes):
           x_min, x_max = strikes.min(), strikes.max()
           if np.isfinite(gamma_flip):
               # Sombreia a região de gamma positiva (dealers dampen).
               ax.axvspan(x_min, gamma_flip, color="#E5F3FF", alpha=0.35,
                          label="Positive Gamma: dealers dampen moves")
               # Sombreia a região de gamma negativa (dealers amplify).
               ax.axvspan(gamma_flip, x_max, color="#FEE2E2", alpha=0.35,
                          label="Negative Gamma: dealers amplify moves")
   
       # Ajusta a escala do eixo Y de forma adaptativa.
       # O objetivo é garantir que o gráfico seja bem dimensionado.
       ymin = float(np.nanmin(gvals)) if len(gvals) else -1.0
       ymax = float(np.nanmax(gvals)) if len(gvals) else  1.0
       if ymin < 0 and ymax > 0:
           lim = max(abs(ymin), abs(ymax))*1.25
           ax.set_ylim(-lim, lim)
       else:
           pad = 0.15*(ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
           ax.set_ylim(ymin - pad, ymax + pad)
   
       # Formata o eixo Y com separadores de milhar.
       ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
       # Define rótulo do eixo X.
       ax.set_xlabel("Strike Price")
       # Define rótulo do eixo Y.
       ax.set_ylabel("Gamma Exposure (USD, millions)")
       # Define título do gráfico.
       ax.set_title("Gamma Exposure by Strike — BOVA11")  # Nota: título menciona PETR4, mas código é para BOVA11; possivelmente um erro.
   
       # Adiciona linhas para paredes de call e put.
       if np.isfinite(call_wall):
           ax.axvline(call_wall, color="#374151", linestyle=":",  lw=1.6,
                       label=f"Call Wall: {call_wall:.2f}")
       if np.isfinite(put_wall):
           ax.axvline(put_wall,  color="#9CA3AF", linestyle="--", lw=1.6,
                       label=f"Put Wall: {put_wall:.2f}")
   
       # Adiciona legenda, grid e marca d'água.
       # O objetivo é tornar o gráfico informativo e profissional.
       ax.legend(loc="upper right", ncol=1, fontsize=9, framealpha=0.95)
       fig.text(0.5, 0.96, "om-qs.com", ha="center", va="center", fontsize=9, alpha=0.7)
       ax.grid(alpha=0.25)
       # Ajusta o layout.
       plt.tight_layout(rect=[0, 0, 1, 0.94])
       # Exibe o gráfico.
       plt.show()
   
       # ------------------------------------------------------------
       # MÉTRICAS ESTENDIDAS DE ESTRUTURA — Interpretação qualitativa
       # ------------------------------------------------------------
       # Imprime cabeçalho da seção estendida.
       print("\n" + "="*75)
       print("EXTENDED MARKET STRUCTURE METRICS — STOCK TRACE-Lite View")
       print("="*75)
   
       # --- Recomputa gamma flip suavizado para consistência
       # Extrai strikes e GEX.
       strikes = gex_by_strike["Strike"].to_numpy(dtype=float)
       gvals   = gex_by_strike["GEX_customer"].to_numpy(dtype=float)
       # Aplica média móvel de 5 pontos.
       smooth  = pd.Series(gvals).rolling(5, center=True, min_periods=1).mean().values
       # Encontra gamma flip na curva mais suavizada.
       gamma_flip = strikes[np.argmin(np.abs(smooth))] if len(strikes) else np.nan
   
       # --- Recap de PCR global
       # Armazena OI de calls e puts.
       calls_oi = total_calls
       puts_oi  = total_puts
       pcr_oi   = pcr_global
   
       # Imprime PCR baseado em OI.
       print(f"Put/Call Ratio (OI):  {pcr_oi:>6.2f}")
       # Classifica o sentimento com base no PCR.
       # O objetivo é fornecer uma interpretação qualitativa do sentimento de mercado.
       if 0.9 <= pcr_oi <= 1.1:
           sentiment = "Neutral"
       elif pcr_oi > 1.1:
           sentiment = "Bearish — put demand dominates"
       else:
           sentiment = "Bullish — call demand dominates"
       # Imprime o sentimento.
       print(f"Sentiment:            {sentiment}")
   
       # --- Interpretação de skew de volatilidade
       print("\nVolatility Skew:")
       # Imprime IV de puts OTM.
       print(f"IV (OTM Puts):   {iv_puts_otm:>6.2f}%")
       # Imprime IV de calls OTM.
       print(f"IV (OTM Calls):  {iv_calls_otm:>6.2f}%")
       # Imprime skew.
       print(f"Skew (Puts−Calls): {iv_skew:>6.2f}%")
   
       # Interpreta o skew qualitativamente.
       if iv_skew > 10:
           print("Interpretation:  Elevated skew — investors hedging downside risk.")
       elif iv_skew < 0:
           print("Interpretation:  Inverted skew — speculative upside bias.")
       else:
           print("Interpretation:  Balanced implied vol surface.")
   
       # --- Análise de gamma flip
       print("\nGamma Flip Analysis:")
       # Imprime gamma flip.
       print(f"Gamma Flip (approx): {gamma_flip:>8.2f}")
       # Imprime spot.
       print(f"Spot:                 {spot:>8.2f}")
   
       # Calcula diferença relativa ao flip.
       if np.isfinite(gamma_flip):
           diff = spot - gamma_flip
           pct  = diff / gamma_flip * 100
           side = "above" if diff > 0 else "below"
           # Imprime posição relativa.
           print(f"Spot is {abs(pct):.2f}% {side} the flip.")
           # Interpreta o impacto dos dealers.
           if diff > 0:
               print("→ Dealers short gamma: market mechanically amplified.")
           else:
               print("→ Dealers long gamma: market mechanically dampened.")
   
       # --- Classificação de regime de mercado
       # Classifica o regime com base na posição relativa ao gamma flip.
       # O objetivo é sugerir estratégias baseadas no regime detectado.
       if np.isfinite(gamma_flip):
           if spot >= gamma_flip * 1.05:
               regime = "HIGH VOLATILITY"
               rationale = "Dealers short gamma, hedging exacerbates moves."
               strategy = "Long gamma, directional or convexity-driven setups."
           elif spot <= gamma_flip * 0.95:
               regime = "LOW VOLATILITY"
               rationale = "Dealers long gamma, hedging absorbs shocks."
               strategy = "Range trading, vol selling, short gamma spreads."
           else:
               regime = "TRANSITION ZONE"
               rationale = "Market near flip — unstable hedging behavior."
               strategy = "Neutral, calendar, or butterfly setups."
       else:
           regime, rationale, strategy = "UNKNOWN", "Gamma Flip not found", "N/A"
   
       # Imprime o regime detectado.
       print("\nMarket Regime:")
       print(f"Detected:     {regime}")
       # Imprime a rationale.
       print(f"Rationale:    {rationale}")
       # Imprime a estratégia recomendada.
       print(f"Recommended:  {strategy}")
   
       # --- Zonas significativas de GEX inferidas
       print("\nSignificant GEX Zones:")
       # Ordena GEX por valor descendente.
       gex_sorted = gex_by_strike.sort_values("GEX_customer", ascending=False)
       # Seleciona top 4 para resistance (gamma positiva).
       resist = gex_sorted.head(4)   
       # Seleciona bottom 4 para support (gamma negativa).
       supports = gex_sorted.tail(4) 
   
       # Imprime zonas de suporte.
       # O objetivo é identificar níveis onde dealers podem amortecer movimentos.
       print("Support Zones (dealers long gamma → cushion):")
       for _, r in supports.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           # Classifica força com base no valor absoluto.
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | {gex_mil:>7.2f}M | {strength}")
   
       # Imprime zonas de resistance.
       # O objetivo é identificar níveis onde movimentos podem ser amplificados.
       print("\nResistance Zones (dealers short gamma → acceleration risk):")
       for _, r in resist.iterrows():
           gex_mil = r["GEX_customer"] / 1e6
           strength = "Strong" if abs(gex_mil) > 200 else "Moderate" if abs(gex_mil) > 100 else "Weak"
           print(f"  Strike {r['Strike']:>8.2f} | +{gex_mil:>7.2f}M | {strength}")
   
       # --- Snapshot de resumo
       # Imprime um resumo final com níveis chave.
       # O objetivo é fornecer uma visão rápida das métricas principais.
       print("\nSummary Snapshot:")
       print(f"Spot:        {spot:,.2f}")
       print(f"Call Wall:   {call_wall:,.2f}")
       print(f"Put Wall:    {put_wall:,.2f}")
       print(f"Gamma Flip:  {gamma_flip:,.2f}")
       print(f"Market Regime: {regime}")
       print("="*75)
       # Imprime as primeiras linhas do DataFrame para verificação.
       print(df.head())
       print("="*75)
       # Imprime as colunas do DataFrame.
       print(df.columns)

async def main():
    # Conecta ao MetaTrader 5 para obter o preço spot atual de PETR4.
    mt5_conn = MT5Connector()
    symbol_info = mt5_conn.get_symbol_info(UNDERLYING)
    spot_price = symbol_info.bid  # Usa o preço de compra como spot
    # Chama a função de análise com o arquivo Excel e o preço spot.
    print(f"Analyzing options data for {UNDERLYING} with spot price {spot_price:.2f}...")
    
    file_path = os.path.join(SCRIPT_DIR, "BOVA11_Options_B3_Format.xlsx")
    await analyze_options(file_path, spot_price)
# ------------------------------------------------------------
# Exemplo de uso (descomente para executar)
# ------------------------------------------------------------
asyncio.run(main())