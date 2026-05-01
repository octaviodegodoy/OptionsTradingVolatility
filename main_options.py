from datetime import datetime
import logging
import asyncio
from constants import (
    ASSET_SYMBOL, DIFF_IV_GARCH_PUTS_THRESHOLD_PCT, GARCH_SAMPLE_SIZE,
    IV_DIFF_THRESHOLD_CALLS, MIN_PUT_IV, STEEP_THRESHOLD, ANNUAL_TRADING_DAYS,
    ACTIVE_STRATEGY, VOLATILITY_SKEW, STRADDLE, PUT_SPREAD, SHORT_CALL_BUTTERFLY,
    STRADDLE_MAX_DELTA_IMBALANCE, STRADDLE_SLEEP_SECONDS,
    PUT_SPREAD_EXPIRY_RANK, PUT_SPREAD_LONG_DELTA_MIN, PUT_SPREAD_LONG_DELTA_MAX,
    PUT_SPREAD_SHORT_DELTA_MIN, PUT_SPREAD_SHORT_DELTA_MAX,
    PUT_SPREAD_MIN_IV_EDGE, PUT_SPREAD_MAX_POSITIONS, PUT_SPREAD_SLEEP_SECONDS,
    SHORT_CALL_BUTTERFLY_EXPIRY_RANK, SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT,
    SHORT_CALL_BUTTERFLY_MIN_IV_EDGE, SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT,
    SHORT_CALL_BUTTERFLY_MAX_POSITIONS, SHORT_CALL_BUTTERFLY_ORDER_SIZE,
    SHORT_CALL_BUTTERFLY_SLEEP_SECONDS,
)
from mt5_connector import MT5Connector
from functions.quant_functions import QuantCalculation
import numpy as np
from utils import Utils


# ================================================================
# Strategies — each one runs its own loop with its own trade logic
# ================================================================

def select_short_call_butterfly_candidate(utils, calls_dict, spot_price, garch_vol):
    call_by_strike = {}
    for call_delta, call_data in calls_dict.items():
        strike = call_data["strike"]
        current = call_by_strike.get(strike)
        if current is None or abs(call_delta - 0.50) < abs(current[0] - 0.50):
            call_by_strike[strike] = (call_delta, call_data)

    if len(call_by_strike) < 3:
        return None

    quote_by_strike = {}
    for strike, (_, call_data) in call_by_strike.items():
        option_info = utils.get_option_info_with_quote(call_data["option_name"])
        if option_info is None or option_info.bid <= 0.0 or option_info.ask <= 0.0:
            continue
        quote_by_strike[strike] = option_info

    available_strikes = sorted(quote_by_strike.keys())
    if len(available_strikes) < 3:
        return None

    best_candidate = None
    available_strikes_set = set(available_strikes)
    for middle_idx in range(1, len(available_strikes) - 1):
        middle_strike = available_strikes[middle_idx]
        body_distance_pct = abs(middle_strike - spot_price) / max(spot_price, 1.0)
        if body_distance_pct > SHORT_CALL_BUTTERFLY_MAX_BODY_DISTANCE_PCT:
            continue

        for lower_strike in available_strikes[:middle_idx]:
            wing_width = middle_strike - lower_strike
            if wing_width <= 0:
                continue

            upper_strike = round(middle_strike + wing_width, 8)
            if upper_strike not in available_strikes_set:
                continue

            lower_delta, lower_data = call_by_strike[lower_strike]
            middle_delta, middle_data = call_by_strike[middle_strike]
            upper_delta, upper_data = call_by_strike[upper_strike]
            lower_quote = quote_by_strike[lower_strike]
            middle_quote = quote_by_strike[middle_strike]
            upper_quote = quote_by_strike[upper_strike]

            net_credit = lower_quote.bid + upper_quote.bid - (2 * middle_quote.ask)
            body_iv_edge = garch_vol - middle_data["iv"]
            wing_richness = ((lower_data["iv"] + upper_data["iv"]) / 2) - middle_data["iv"]
            score = (
                body_iv_edge
                + wing_richness
                + net_credit
                - (body_distance_pct * 100)
            )

            candidate = {
                "score": score,
                "body_iv_edge": body_iv_edge,
                "wing_richness": wing_richness,
                "net_credit": net_credit,
                "body_distance_pct": body_distance_pct,
                "wing_width": wing_width,
                "lower_symbol": lower_data["option_name"],
                "middle_symbol": middle_data["option_name"],
                "upper_symbol": upper_data["option_name"],
                "lower_strike": lower_strike,
                "middle_strike": middle_strike,
                "upper_strike": upper_strike,
                "lower_delta": lower_delta,
                "middle_delta": middle_delta,
                "upper_delta": upper_delta,
                "lower_iv": lower_data["iv"],
                "middle_iv": middle_data["iv"],
                "upper_iv": upper_data["iv"],
                "lower_bid": lower_quote.bid,
                "middle_ask": middle_quote.ask,
                "upper_bid": upper_quote.bid,
            }

            if best_candidate is None or candidate["score"] > best_candidate["score"]:
                best_candidate = candidate

    return best_candidate

async def strategy_volatility_skew(mt5_conn, quant_calc, utils, logger):
    """
    VOLATILITY_SKEW strategy:
    - Long ATM puts when IV is cheap vs GARCH or put skew is steep
    - Call vertical spreads when IV difference between strikes is compressed
    """
    asset = ASSET_SYMBOL[0]

    selected_asset = mt5_conn.symbol_select(asset, True)
    if not selected_asset:
        logger.error(f"Failed to select {asset}")
        return

    # Wait until MT5 streams live tick data for the underlying
    for _ in range(10):
        tick = mt5_conn.get_mt5_connector().symbol_info_tick(asset)
        if tick is not None and tick.bid > 0 and tick.ask > 0:
            break
        logger.info(f"Waiting for {asset} tick data after symbol_select...")
        await asyncio.sleep(1)
    else:
        logger.error(f"{asset} still has no tick data after 10 s – check Market Watch")
        return

    symbol_info = mt5_conn.get_symbol_info(asset)  # initial fetch for chain loading
    chain_options = mt5_conn.get_option_names_by_expiration_time(asset)
    logger.info(f"Options Chain for {asset} retrieved {len(chain_options.values())} options.")
    expiration_time = list(chain_options.keys())[0]

    while selected_asset:
       mt5_conn.symbol_select(asset, True)  # ensure symbol stays in Market Watch
       symbol_info = mt5_conn.get_symbol_info(asset)
       if symbol_info.bid == 0.0 or symbol_info.ask == 0.0:
           logger.warning(f"Underlying {asset} bid/ask is 0.0, waiting for market data...")
           await asyncio.sleep(5)
           continue

       spot_prices_data = mt5_conn.get_data(asset, mt5_conn.get_mt5_connector().TIMEFRAME_D1, GARCH_SAMPLE_SIZE, 0)["close"].values
       garch_vol = quant_calc.agarch_estimation(spot_prices_data)*100
       logger.info(f"GARCH Volatility : {garch_vol:.2f}%")

       logger.info(f"Selected Expiration Time: {datetime.fromtimestamp(expiration_time)}")
       calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)

       if not calls_dict or not puts_dict:
           logger.warning("calls_dict or puts_dict is empty, skipping iteration")
           await asyncio.sleep(15)
           continue
   
       call_iv_dict = {k: v['iv'] for k, v in calls_dict.items()}
       put_iv_dict = {k: v['iv'] for k, v in puts_dict.items()}
   
       print(f"Put IV dict: {put_iv_dict}")
       print(f"Call IV dict: {call_iv_dict}")
   
       real_delta_calls = np.array(list(call_iv_dict.keys()))
       real_delta_puts  = np.array(list(put_iv_dict.keys()))
       real_iv_calls = np.array(list(call_iv_dict.values()))
       real_iv_puts  = np.array(list(put_iv_dict.values()))

       result = quant_calc.are_puts_steeper(real_delta_puts, real_iv_puts, real_delta_calls, real_iv_calls, STEEP_THRESHOLD)
       
       atm_price = (symbol_info.bid + symbol_info.ask) / 2
       print(f"ATM strike price for {asset} is approximately {atm_price}")
   
       put_strikes_and_ivs = [(v['strike'], v['iv']) for v in puts_dict.values()]
       sorted_strikes = sorted({v['strike'] for v in puts_dict.values()})
       print(f"All put strikes : {sorted_strikes}")
       atm_strike = min(sorted_strikes, key=lambda x: abs(x - atm_price))
   
       print(f"ATM strike determined from available strikes: {atm_strike}")
       atm_idx = sorted_strikes.index(atm_strike)
       atm_strikes = sorted_strikes[max(0, atm_idx - 1):min(len(sorted_strikes), atm_idx + 2)]
       print(f"ATM strikes: {atm_strikes}")
   
       atm_ivs = [(iv, strike) for strike, iv in put_strikes_and_ivs if strike in atm_strikes]
       print(f"ATM IVs: {atm_ivs}")
   
       min_iv_strike = min(atm_ivs, key=lambda x: x[0])
       put_atm_delta = next((k for k, v in puts_dict.items() if v['strike'] == min_iv_strike[1]), None)
       print(f"Minimum IV at ATM strikes: {min_iv_strike[0]:.2f}% at strike {min_iv_strike[1]} with delta {put_atm_delta:.2f}")
       iv_garch_diff_pct = min_iv_strike[0] - garch_vol
       print(f"Strike with minimum IV: {min_iv_strike[1]}, IV: {min_iv_strike[0]} and GARCH volatility: {garch_vol:.2f}% and puts steeper? {result} and min IV at ATM puts is {iv_garch_diff_pct:.2f}% different from GARCH volatility (threshold was {DIFF_IV_GARCH_PUTS_THRESHOLD_PCT} pp)")
       put_name_min_iv = next((v['option_name'] for v in puts_dict.values() if v['strike'] == min_iv_strike[1]), None)
      
       put_condition = (iv_garch_diff_pct <= DIFF_IV_GARCH_PUTS_THRESHOLD_PCT or result) and min_iv_strike[0] >= MIN_PUT_IV

       print(f"Put condition for buying: {put_condition} (IV difference from GARCH: {iv_garch_diff_pct:.2f}% and puts steeper? {result} and ATM IV {min_iv_strike[0]:.2f}% >= MIN_PUT_IV {MIN_PUT_IV}%)")       

       # ── Scan all eligible call pairs for minimum IV difference ──
       # Filter calls by delta range (0.25 – 0.75)
       liquid_calls = {
           delta: data for delta, data in calls_dict.items()
           if 0.25 < delta < 0.75
       }
       liquid_deltas = sorted(liquid_calls.keys())
       print(f"Eligible calls: {[(d, liquid_calls[d]['strike'], liquid_calls[d]['session_volume']) for d in liquid_deltas]}")

       # Compute mean and std of liquid call deltas for the 2-std distance filter
       best_pair = None  # (sell_delta, buy_delta, iv_diff)

       if len(liquid_deltas) >= 2:
           avg_delta = sum(liquid_deltas) / len(liquid_deltas)
           std_delta = (sum((d - avg_delta) ** 2 for d in liquid_deltas) / len(liquid_deltas)) ** 0.5
           required_gap = 2 * std_delta
           print(f"Call delta stats: mean={avg_delta:.4f}, std={std_delta:.4f}, required gap (2σ)={required_gap:.4f}")

           # buy = upper delta (higher), sell = lower delta (lower)
           for i, sell_delta_cand in enumerate(liquid_deltas):
               for buy_delta_cand in liquid_deltas[i + 1:]:
                   delta_gap = buy_delta_cand - sell_delta_cand
                   if delta_gap < required_gap:
                       continue  # too close, skip
                   sell_iv = liquid_calls[sell_delta_cand]['iv']
                   buy_iv  = liquid_calls[buy_delta_cand]['iv']
                   pair_diff = sell_iv - buy_iv  # want sell IV minimised vs buy IV
                   if best_pair is None or pair_diff < best_pair[2]:
                       best_pair = (sell_delta_cand, buy_delta_cand, pair_diff)

       if best_pair is not None:
           sell_delta, buy_delta, iv_diff = best_pair
           iv_sell = liquid_calls[sell_delta]['iv']
           iv_buy  = liquid_calls[buy_delta]['iv']
           call_sell = liquid_calls[sell_delta]['option_name']
           call_buy  = liquid_calls[buy_delta]['option_name']
           print(f"Best call pair → Sell {call_sell} (delta {sell_delta}, IV {iv_sell:.2f}%, strike {liquid_calls[sell_delta]['strike']}) "
                 f"| Buy {call_buy} (delta {buy_delta}, IV {iv_buy:.2f}%, strike {liquid_calls[buy_delta]['strike']}) "
                 f"| IV diff {iv_diff:.2f}%")
       else:
           iv_diff = None
           print("No eligible call pairs found for spread")

       orders_type = [mt5_conn.ORDER_TYPE_BUY, mt5_conn.ORDER_TYPE_SELL]

       call_condition = iv_diff is not None and iv_diff <= IV_DIFF_THRESHOLD_CALLS

       min_amount = 100
       put_atm_delta = abs(put_atm_delta) if put_atm_delta is not None else 0.0
       put_amount = float(round(min_amount/put_atm_delta/min_amount)*min_amount if put_atm_delta is not None and put_atm_delta != 0 else min_amount)
       call_delta = buy_delta - sell_delta if best_pair is not None else 0  # buy(upper) - sell(lower)
       call_amount = float(round(min_amount/call_delta/min_amount)*min_amount if call_delta != 0 else min_amount)
           
       puts_positions_total = utils.put_options_count()
       print(f"Current open put positions: {puts_positions_total}")
       call_positions_total = utils.call_options_count()
       print(f"Current open call positions: {call_positions_total}")

       put_order_allowed = puts_positions_total < 2 and put_condition
       call_order_allowed = call_positions_total < 6 and call_condition

       print(f"Checking order for put option {put_name_min_iv} with amount {put_amount} based on ATM put delta {put_atm_delta:.2f} steep {result} and IV difference from GARCH {iv_garch_diff_pct:.2f}%") 

       if put_order_allowed:          
          logger.info(f"Puts are steeper than calls with a slope difference of at least {STEEP_THRESHOLD} pp/delta.")
          symbol_info = mt5_conn.get_symbol_info(asset)
          mt5_conn.place_order(put_name_min_iv,MT5Connector.ORDER_TYPE_BUY, put_amount, symbol_info.ask, 10, str(min_iv_strike[0]))
       else:
          logger.info(f"Condition not met for put condition: {put_condition} (IV difference from GARCH: {iv_garch_diff_pct:.2f}% and puts steeper? {result})")

       if call_order_allowed:
            print(f"Placing orders for call spread: Buy {call_buy} and Sell {call_sell} | IV diff {f'{iv_diff:.2f}' if iv_diff is not None else 'N/A'}% <= {IV_DIFF_THRESHOLD_CALLS}%")
            mt5_conn.place_order_vertical(call_buy, call_sell, orders_type, call_amount, iv_buy, iv_sell)
       else:
            logger.info(f"Condition not met for call condition: {call_condition} diff threshold is {IV_DIFF_THRESHOLD_CALLS} (IV diff: {f'{iv_diff:.2f}' if iv_diff is not None else 'N/A'}%) for call spread")
            
       await asyncio.sleep(25)    


async def strategy_straddle(mt5_conn, quant_calc, utils, logger):
    """
    STRADDLE strategy:
    - Buy call + buy put on the same strike
    - Select pair with delta-neutral score as close to zero as possible
      where score = |call_delta + put_delta|
    """
    asset = ASSET_SYMBOL[0]

    selected_asset = mt5_conn.symbol_select(asset, True)
    if not selected_asset:
        logger.error(f"Failed to select {asset}")
        return

    for _ in range(10):
        tick = mt5_conn.get_mt5_connector().symbol_info_tick(asset)
        if tick is not None and tick.bid > 0 and tick.ask > 0:
            break
        logger.info(f"Waiting for {asset} tick data after symbol_select...")
        await asyncio.sleep(1)
    else:
        logger.error(f"{asset} still has no tick data after 10 s – check Market Watch")
        return

    last_rebalance_spot = None
    first_delta_neutral_entry_done = False

    while selected_asset:
        mt5_conn.symbol_select(asset, True)
        symbol_info = mt5_conn.get_symbol_info(asset)
        if symbol_info is None or symbol_info.bid == 0.0 or symbol_info.ask == 0.0:
            logger.warning(f"Underlying {asset} bid/ask is invalid, waiting for market data...")
            await asyncio.sleep(5)
            continue

        spot_prices_data = mt5_conn.get_data(
            asset,
            mt5_conn.get_mt5_connector().TIMEFRAME_D1,
            GARCH_SAMPLE_SIZE,
            0,
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        garch_vol_change = garch_vol / np.sqrt(ANNUAL_TRADING_DAYS)
        logger.info(f"GARCH Volatility for STRADDLE: {garch_vol:.2f}% (vol change={garch_vol_change:.2f}% using sqrt({ANNUAL_TRADING_DAYS}))")

        chain_options = mt5_conn.get_option_names_by_expiration_time(asset)
        if not chain_options:
            logger.warning("No option chain returned for selected expiration")
            await asyncio.sleep(STRADDLE_SLEEP_SECONDS)
            continue

        expiration_time = next(iter(chain_options.keys()))
        logger.info(f"Selected Expiration Time: {datetime.fromtimestamp(expiration_time)}")

        calls_dict, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not calls_dict or not puts_dict:
            logger.warning("calls_dict or puts_dict is empty, skipping iteration")
            await asyncio.sleep(STRADDLE_SLEEP_SECONDS)
            continue

        # Build strike maps to enforce true straddles (same strike call+put)
        call_by_strike = {}
        for call_delta, call_data in calls_dict.items():
            call_by_strike[call_data['strike']] = (call_delta, call_data)

        put_by_strike = {}
        for put_delta, put_data in puts_dict.items():
            put_by_strike[put_data['strike']] = (put_delta, put_data)

        common_strikes = sorted(set(call_by_strike.keys()) & set(put_by_strike.keys()))
        if not common_strikes:
            logger.warning("No common strikes between calls and puts for straddle")
            await asyncio.sleep(STRADDLE_SLEEP_SECONDS)
            continue

        logger.info(f"STRADDLE analyzed common strikes ({len(common_strikes)}): {common_strikes}")

        atm_price = (symbol_info.bid + symbol_info.ask) / 2

        # Pick the 2 nearest ATM common strikes, then choose the pair with minimum net IV
        atm_sorted_strikes = sorted(common_strikes, key=lambda s: abs(s - atm_price))[:2]
        logger.info(f"STRADDLE 2 nearest ATM common strikes: {atm_sorted_strikes} (spot={atm_price:.2f})")

        best_pair = None
        for strike in atm_sorted_strikes:
            call_delta, call_data = call_by_strike[strike]
            put_delta, put_data = put_by_strike[strike]
            net_iv = (call_data['iv'] + put_data['iv']) / 2
            neutrality_score = abs(call_delta + put_delta)
            candidate = (net_iv, neutrality_score, strike, call_delta, put_delta, call_data, put_data)
            logger.info(
                f"  strike={strike}: call_iv={call_data['iv']:.2f}%, put_iv={put_data['iv']:.2f}%, "
                f"net_iv={net_iv:.2f}%, |Δc+Δp|={neutrality_score:.4f}"
            )
            if best_pair is None or candidate < best_pair:
                best_pair = candidate

        _, neutrality_score, strike, call_delta, put_delta, call_data, put_data = best_pair
        call_symbol = call_data['option_name']
        put_symbol = put_data['option_name']
        call_iv = call_data['iv']
        put_iv = put_data['iv']
        straddle_net_iv = (call_iv + put_iv) / 2
        straddle_vol_change = straddle_net_iv / np.sqrt(ANNUAL_TRADING_DAYS)

        logger.info(
            f"Best straddle pair: strike={strike}, call={call_symbol} (Δ={call_delta:.2f}, IV={call_iv:.2f}%), "
            f"put={put_symbol} (Δ={put_delta:.2f}, IV={put_iv:.2f}%), neutrality |Δc+Δp|={neutrality_score:.4f}, "
            f"net IV={straddle_net_iv:.2f}% (vol change={straddle_vol_change:.2f}% using sqrt({ANNUAL_TRADING_DAYS}))"
        )

        puts_positions_total = utils.put_options_count()
        call_positions_total = utils.call_options_count()
        has_open_straddle = puts_positions_total > 0 and call_positions_total > 0

        if first_delta_neutral_entry_done and has_open_straddle and last_rebalance_spot is None:
            last_rebalance_spot = atm_price
            logger.info(f"Initialized STRADDLE rebalance anchor at spot={last_rebalance_spot:.2f}")

        rebalance_due = False
        spot_move_pct = 0.0
        if first_delta_neutral_entry_done and has_open_straddle and last_rebalance_spot and last_rebalance_spot > 0:
            spot_move_pct = abs((atm_price - last_rebalance_spot) / last_rebalance_spot) * 100
            rebalance_due = spot_move_pct >= straddle_vol_change

        if first_delta_neutral_entry_done and has_open_straddle and not rebalance_due:
            logger.info(
                f"Open STRADDLE held: spot move={spot_move_pct:.2f}% < trigger={straddle_vol_change:.2f}% "
                f"(rebalance anchor={last_rebalance_spot:.2f}, current spot={atm_price:.2f})"
            )
            await asyncio.sleep(STRADDLE_SLEEP_SECONDS)
            continue

        if first_delta_neutral_entry_done and has_open_straddle and rebalance_due:
            logger.info(
                f"Rebalancing STRADDLE: spot move={spot_move_pct:.2f}% >= trigger={straddle_vol_change:.2f}% "
                f"from anchor={last_rebalance_spot:.2f} to spot={atm_price:.2f}. Closing strategy positions to re-hedge delta neutral."
            )
            mt5_conn.close_all_positions()
            await asyncio.sleep(1)
            puts_positions_total = utils.put_options_count()
            call_positions_total = utils.call_options_count()

        delta_neutral_condition = neutrality_score <= STRADDLE_MAX_DELTA_IMBALANCE
        first_entry_vol_garch_condition = True
        if puts_positions_total == 0 and call_positions_total == 0 and last_rebalance_spot is None:
            first_entry_vol_garch_condition = straddle_net_iv <= garch_vol
        straddle_allowed = (
            delta_neutral_condition and
            first_entry_vol_garch_condition and
            puts_positions_total == 0 and
            call_positions_total == 0
        )

        if not straddle_allowed:
            logger.info(
                f"STRADDLE not placed: allowed={straddle_allowed}, |Δc+Δp|={neutrality_score:.4f}, "
                f"threshold={STRADDLE_MAX_DELTA_IMBALANCE}, net_iv={straddle_net_iv:.2f}%, garch={garch_vol:.2f}%, "
                f"first_entry_vol_garch={first_entry_vol_garch_condition}, "
                f"open puts={puts_positions_total}, open calls={call_positions_total}"
            )
            await asyncio.sleep(STRADDLE_SLEEP_SECONDS)
            continue

        call_symbol_info = mt5_conn.get_symbol_info(call_symbol)
        put_symbol_info = mt5_conn.get_symbol_info(put_symbol)
        if (
            call_symbol_info is None or put_symbol_info is None or
            call_symbol_info.ask <= 0 or put_symbol_info.ask <= 0
        ):
            logger.warning("Selected straddle legs have invalid ask price; skipping")
            await asyncio.sleep(STRADDLE_SLEEP_SECONDS)
            continue

        straddle_amount = 100.0
        logger.info(
            f"Placing STRADDLE: BUY {call_symbol} @ {call_symbol_info.ask:.4f} and "
            f"BUY {put_symbol} @ {put_symbol_info.ask:.4f} volume={straddle_amount}"
        )

        mt5_conn.place_order(
            call_symbol,
            MT5Connector.ORDER_TYPE_BUY,
            straddle_amount,
            call_symbol_info.ask,
            10,
            f"SC d={call_delta:.2f} iv={call_iv:.1f}",
        )
        mt5_conn.place_order(
            put_symbol,
            MT5Connector.ORDER_TYPE_BUY,
            straddle_amount,
            put_symbol_info.ask,
            10,
            f"SP d={put_delta:.2f} iv={put_iv:.1f}",
        )

        first_delta_neutral_entry_done = True
        last_rebalance_spot = atm_price

        await asyncio.sleep(STRADDLE_SLEEP_SECONDS)


async def strategy_put_spread(mt5_conn, quant_calc, utils, logger):
    """
    PUT_SPREAD strategy — bearish put spread (debit spread):
    - Buy a moderately OTM put (long leg, higher strike, closer to ATM)
    - Sell a further OTM put (short leg, lower strike, for premium offset)
    - Enter when long-put IV is sufficiently below GARCH volatility
      (cheap premium relative to expected realised vol = edge for the buyer)
    - Expiry is selected by PUT_SPREAD_EXPIRY_RANK (1=next, 2=second, 3=third)
    """
    asset = ASSET_SYMBOL[0]

    selected_asset = mt5_conn.symbol_select(asset, True)
    if not selected_asset:
        logger.error(f"Failed to select {asset}")
        return

    for _ in range(10):
        tick = mt5_conn.get_mt5_connector().symbol_info_tick(asset)
        if tick is not None and tick.bid > 0 and tick.ask > 0:
            break
        logger.info(f"Waiting for {asset} tick data after symbol_select...")
        await asyncio.sleep(1)
    else:
        logger.error(f"{asset} still has no tick data after 10 s – check Market Watch")
        return

    while True:
        mt5_conn.symbol_select(asset, True)
        symbol_info = mt5_conn.get_symbol_info(asset)
        if symbol_info is None or symbol_info.bid == 0.0 or symbol_info.ask == 0.0:
            logger.warning(f"Underlying {asset} bid/ask is invalid, waiting...")
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        # ── GARCH vol ──────────────────────────────────────────────────────────
        spot_prices_data = mt5_conn.get_data(
            asset,
            mt5_conn.get_mt5_connector().TIMEFRAME_D1,
            GARCH_SAMPLE_SIZE,
            0,
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        logger.info(f"PUT_SPREAD | GARCH vol: {garch_vol:.2f}%")

        # ── Option chain for the chosen expiry rank ────────────────────────────
        chain_options = mt5_conn.get_option_names_by_expiration_time(
            asset, expiry_rank_override=PUT_SPREAD_EXPIRY_RANK
        )
        if not chain_options:
            logger.warning("No option chain returned for selected expiry rank")
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        expiration_time = next(iter(chain_options.keys()))
        logger.info(f"PUT_SPREAD | Expiry: {datetime.fromtimestamp(expiration_time)} (rank {PUT_SPREAD_EXPIRY_RANK})")

        _, puts_dict = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not puts_dict:
            logger.warning("puts_dict is empty, skipping iteration")
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        # ── Select legs by delta ranges (use abs delta for puts) ────────────────
        long_candidates = {
            d: v for d, v in puts_dict.items()
            if PUT_SPREAD_LONG_DELTA_MIN <= abs(d) <= PUT_SPREAD_LONG_DELTA_MAX
        }
        short_candidates = {
            d: v for d, v in puts_dict.items()
            if PUT_SPREAD_SHORT_DELTA_MIN <= abs(d) <= PUT_SPREAD_SHORT_DELTA_MAX
        }

        if not long_candidates or not short_candidates:
            logger.info(
                f"PUT_SPREAD | Not enough eligible puts — "
                f"long candidates: {len(long_candidates)}, short candidates: {len(short_candidates)}"
            )
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        # Best pair: long leg must have higher strike than short leg.
        # Maximise (garch_vol - long_iv) i.e. cheapness of the long leg,
        # plus (short_iv - long_iv) i.e. skew premium collected on short leg.
        best = None  # (score, iv_edge, long_delta, short_delta, long_data, short_data)
        for l_delta, l_data in long_candidates.items():
            for s_delta, s_data in short_candidates.items():
                if l_data['strike'] <= s_data['strike']:
                    continue  # long leg must be the higher strike
                iv_edge = garch_vol - l_data['iv']          # positive = long IV cheap vs GARCH
                spread_skew = s_data['iv'] - l_data['iv']   # skew benefit on short leg
                score = iv_edge + spread_skew
                if best is None or score > best[0]:
                    best = (score, iv_edge, l_delta, s_delta, l_data, s_data)

        if best is None:
            logger.info("PUT_SPREAD | No valid bearish put spread pair found")
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        _, iv_edge, long_delta, short_delta, long_data, short_data = best
        long_symbol  = long_data['option_name']
        short_symbol = short_data['option_name']
        long_iv      = long_data['iv']
        short_iv     = short_data['iv']
        long_strike  = long_data['strike']
        short_strike = short_data['strike']

        logger.info(
            f"PUT_SPREAD | Best spread → "
            f"BUY  {long_symbol}  strike={long_strike}  Δ={long_delta:.2f}  IV={long_iv:.2f}% | "
            f"SELL {short_symbol} strike={short_strike} Δ={short_delta:.2f} IV={short_iv:.2f}% | "
            f"IV edge vs GARCH={iv_edge:.2f}pp (threshold={PUT_SPREAD_MIN_IV_EDGE}pp)"
        )

        # ── Entry guard ─────────────────────────────────────────────────────────
        open_puts = utils.put_options_count()
        iv_edge_condition = iv_edge >= PUT_SPREAD_MIN_IV_EDGE
        position_room     = open_puts < PUT_SPREAD_MAX_POSITIONS * 2  # 2 legs per spread
        order_allowed     = iv_edge_condition and position_room

        if not order_allowed:
            logger.info(
                f"PUT_SPREAD | Entry blocked — iv_edge_ok={iv_edge_condition} "
                f"({iv_edge:.2f}pp >= {PUT_SPREAD_MIN_IV_EDGE}pp), "
                f"position_room={position_room} (open puts={open_puts} < {PUT_SPREAD_MAX_POSITIONS * 2})"
            )
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        # Validate live quotes for both legs
        long_info  = mt5_conn.get_symbol_info(long_symbol)
        short_info = mt5_conn.get_symbol_info(short_symbol)
        if (
            long_info is None or short_info is None or
            long_info.ask <= 0 or short_info.bid <= 0
        ):
            logger.warning("PUT_SPREAD | One or both legs have no valid quote; skipping")
            await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)
            continue

        spread_amount = 100.0
        logger.info(
            f"PUT_SPREAD | Placing spread: "
            f"BUY  {long_symbol}  @ {long_info.ask:.4f} | "
            f"SELL {short_symbol} @ {short_info.bid:.4f} "
            f"volume={spread_amount}"
        )
        mt5_conn.place_order(
            long_symbol,
            MT5Connector.ORDER_TYPE_BUY,
            spread_amount,
            long_info.ask,
            10,
            f"BP-long  d={long_delta:.2f} iv={long_iv:.1f}",
        )
        mt5_conn.place_order(
            short_symbol,
            MT5Connector.ORDER_TYPE_SELL,
            spread_amount,
            short_info.bid,
            10,
            f"BP-short d={short_delta:.2f} iv={short_iv:.1f}",
        )

        await asyncio.sleep(PUT_SPREAD_SLEEP_SECONDS)


async def strategy_short_call_butterfly(mt5_conn, quant_calc, utils, logger):
    """
    SHORT_CALL_BUTTERFLY strategy:
    - Sell 1 lower-strike call
    - Buy 2 ATM/near-ATM calls at the body strike
    - Sell 1 higher-strike call
    - Require symmetric call wings around the body strike
    - Prefer setups where the body IV is cheap vs GARCH and the structure can be
      opened for at least a flat-to-credit cashflow using live quotes
    """
    asset = ASSET_SYMBOL[0]

    selected_asset = mt5_conn.symbol_select(asset, True)
    if not selected_asset:
        logger.error(f"Failed to select {asset}")
        return

    for _ in range(10):
        tick = mt5_conn.get_mt5_connector().symbol_info_tick(asset)
        if tick is not None and tick.bid > 0 and tick.ask > 0:
            break
        logger.info(f"Waiting for {asset} tick data after symbol_select...")
        await asyncio.sleep(1)
    else:
        logger.error(f"{asset} still has no tick data after 10 s – check Market Watch")
        return

    while True:
        mt5_conn.symbol_select(asset, True)
        symbol_info = mt5_conn.get_symbol_info(asset)
        if symbol_info is None or symbol_info.bid == 0.0 or symbol_info.ask == 0.0:
            logger.warning(f"Underlying {asset} bid/ask is invalid, waiting...")
            await asyncio.sleep(SHORT_CALL_BUTTERFLY_SLEEP_SECONDS)
            continue

        spot_prices_data = mt5_conn.get_data(
            asset,
            mt5_conn.get_mt5_connector().TIMEFRAME_D1,
            GARCH_SAMPLE_SIZE,
            0,
        )["close"].values
        garch_vol = quant_calc.agarch_estimation(spot_prices_data) * 100
        logger.info(f"SHORT_CALL_BUTTERFLY | GARCH vol: {garch_vol:.2f}%")

        chain_options = mt5_conn.get_option_names_by_expiration_time(
            asset, expiry_rank_override=SHORT_CALL_BUTTERFLY_EXPIRY_RANK
        )
        if not chain_options:
            logger.warning("SHORT_CALL_BUTTERFLY | No option chain returned for selected expiry rank")
            await asyncio.sleep(SHORT_CALL_BUTTERFLY_SLEEP_SECONDS)
            continue

        expiration_time = next(iter(chain_options.keys()))
        logger.info(
            f"SHORT_CALL_BUTTERFLY | Expiry: {datetime.fromtimestamp(expiration_time)} "
            f"(rank {SHORT_CALL_BUTTERFLY_EXPIRY_RANK})"
        )

        calls_dict, _ = utils.get_calls_and_puts_data(chain_options, symbol_info)
        if not calls_dict:
            logger.warning("SHORT_CALL_BUTTERFLY | calls_dict is empty, skipping iteration")
            await asyncio.sleep(SHORT_CALL_BUTTERFLY_SLEEP_SECONDS)
            continue

        atm_price = (symbol_info.bid + symbol_info.ask) / 2
        candidate = select_short_call_butterfly_candidate(utils, calls_dict, atm_price, garch_vol)
        if candidate is None:
            logger.info("SHORT_CALL_BUTTERFLY | No symmetric butterfly candidate found")
            await asyncio.sleep(SHORT_CALL_BUTTERFLY_SLEEP_SECONDS)
            continue

        logger.info(
            f"SHORT_CALL_BUTTERFLY | Best butterfly → "
            f"SELL {candidate['lower_symbol']} strike={candidate['lower_strike']} Δ={candidate['lower_delta']:.2f} IV={candidate['lower_iv']:.2f}% | "
            f"BUY 2x {candidate['middle_symbol']} strike={candidate['middle_strike']} Δ={candidate['middle_delta']:.2f} IV={candidate['middle_iv']:.2f}% | "
            f"SELL {candidate['upper_symbol']} strike={candidate['upper_strike']} Δ={candidate['upper_delta']:.2f} IV={candidate['upper_iv']:.2f}% | "
            f"wing={candidate['wing_width']:.2f} net_credit={candidate['net_credit']:.4f} "
            f"body_iv_edge={candidate['body_iv_edge']:.2f}pp"
        )

        open_calls = utils.call_options_count()
        iv_edge_condition = candidate["body_iv_edge"] >= SHORT_CALL_BUTTERFLY_MIN_IV_EDGE
        credit_condition = candidate["net_credit"] >= SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT
        position_room = open_calls < SHORT_CALL_BUTTERFLY_MAX_POSITIONS * 3
        order_allowed = iv_edge_condition and credit_condition and position_room

        if not order_allowed:
            logger.info(
                f"SHORT_CALL_BUTTERFLY | Entry blocked — "
                f"iv_edge_ok={iv_edge_condition} ({candidate['body_iv_edge']:.2f}pp >= {SHORT_CALL_BUTTERFLY_MIN_IV_EDGE}pp), "
                f"credit_ok={credit_condition} ({candidate['net_credit']:.4f} >= {SHORT_CALL_BUTTERFLY_MIN_NET_CREDIT:.4f}), "
                f"position_room={position_room} (open calls={open_calls} < {SHORT_CALL_BUTTERFLY_MAX_POSITIONS * 3})"
            )
            await asyncio.sleep(SHORT_CALL_BUTTERFLY_SLEEP_SECONDS)
            continue

        logger.info(
            f"SHORT_CALL_BUTTERFLY | Placing structure: "
            f"SELL {candidate['lower_symbol']} @ {candidate['lower_bid']:.4f} | "
            f"BUY 2x {candidate['middle_symbol']} @ {candidate['middle_ask']:.4f} | "
            f"SELL {candidate['upper_symbol']} @ {candidate['upper_bid']:.4f} "
            f"base_volume={SHORT_CALL_BUTTERFLY_ORDER_SIZE}"
        )

        mt5_conn.place_order(
            candidate["lower_symbol"],
            MT5Connector.ORDER_TYPE_SELL,
            SHORT_CALL_BUTTERFLY_ORDER_SIZE,
            candidate["lower_bid"],
            10,
            f"SCBF-low d={candidate['lower_delta']:.2f} iv={candidate['lower_iv']:.1f}",
        )
        mt5_conn.place_order(
            candidate["middle_symbol"],
            MT5Connector.ORDER_TYPE_BUY,
            SHORT_CALL_BUTTERFLY_ORDER_SIZE * 2,
            candidate["middle_ask"],
            10,
            f"SCBF-mid d={candidate['middle_delta']:.2f} iv={candidate['middle_iv']:.1f}",
        )
        mt5_conn.place_order(
            candidate["upper_symbol"],
            MT5Connector.ORDER_TYPE_SELL,
            SHORT_CALL_BUTTERFLY_ORDER_SIZE,
            candidate["upper_bid"],
            10,
            f"SCBF-up d={candidate['upper_delta']:.2f} iv={candidate['upper_iv']:.1f}",
        )

        await asyncio.sleep(SHORT_CALL_BUTTERFLY_SLEEP_SECONDS)

STRATEGY_MAP = {
    VOLATILITY_SKEW: strategy_volatility_skew,
    STRADDLE: strategy_straddle,
    PUT_SPREAD: strategy_put_spread,
    SHORT_CALL_BUTTERFLY: strategy_short_call_butterfly,
}


async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    mt5_conn = MT5Connector()
    utils = Utils()
    quant_calc = QuantCalculation()

    logger.info(f">>> Active strategy: {ACTIVE_STRATEGY}")

    handler = STRATEGY_MAP.get(ACTIVE_STRATEGY)
    if handler is None:
        logger.error(f"Unknown strategy: {ACTIVE_STRATEGY}. Available: {list(STRATEGY_MAP.keys())}")
        return

    await handler(mt5_conn, quant_calc, utils, logger)


asyncio.run(main())
