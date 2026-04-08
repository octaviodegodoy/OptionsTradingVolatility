from datetime import datetime
import logging
import asyncio
from constants import (
    ASSET_SYMBOL, DIFF_IV_GARCH_PUTS_THRESHOLD_PCT, GARCH_SAMPLE_SIZE,
    IV_DIFF_THRESHOLD_CALLS, MIN_PUT_IV, STEEP_THRESHOLD, ANNUAL_TRADING_DAYS,
    ACTIVE_STRATEGY, VOLATILITY_SKEW, STRADDLE,
    STRADDLE_MAX_DELTA_IMBALANCE, STRADDLE_SLEEP_SECONDS,
)
from mt5_connector import MT5Connector
from functions.quant_functions import QuantCalculation
import numpy as np
from utils import Utils


# ================================================================
# Strategies — each one runs its own loop with its own trade logic
# ================================================================

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


# ── Strategy dispatcher ──────────────────────────────────────
STRATEGY_MAP = {
    VOLATILITY_SKEW: strategy_volatility_skew,
    STRADDLE: strategy_straddle,
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