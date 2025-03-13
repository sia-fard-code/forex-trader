# numba_functions.py

from numba import njit, prange
import numpy as np
from numba import config
# config.DEBUG_JIT = True
# config.ENABLE_PARALLEL_DIAGNOSTICS = 1

# @njit(parallel=True)
def update_positions_numba_parallel(positions, current_bid, current_ask, tick_id, slippage, commission_per_lot, profit_threshold=None, loss_threshold=None, trailing_pct=None, closing_method=1):
    """
    Parallelized function to update positions based on current bid and ask prices and PnL thresholds.
    Args:
        positions: Array of position data (structured array).
        current_bid: Current bid price.
        current_ask: Current ask price.
        tick_id: Current tick identifier.
        profit_threshold: Profit threshold for closing positions.
        loss_threshold: Loss threshold for closing positions.
        slippage: Slippage applied to bid/ask prices.
        commission_per_lot: Commission per lot for the trade.
        closing_method: The closing method to use ('stopLoss' or 'Trailing').
    """
    n = len(positions)

    for i in prange(n):
        if positions[i]['close_id'] == -1:  # Only process open positions
            direction = positions[i]['direction']
            # direction *= -1
            open_price = positions[i]['open_price']
            volume = 1*positions[i]['volume']
            closing_method = positions[i]['closing_method'] #if positions[i]['closing_method'] is not None else 1
            profit_threshold = positions[i]['profit_threshold'] if profit_threshold is None else profit_threshold
            loss_threshold = positions[i]['loss_threshold'] if loss_threshold is None else loss_threshold
            trailing_pct = positions[i]['trailing_pct'] if trailing_pct is not None else 0.05
            # Determine the current price based on direction
            if direction == 1:  # Long position
                adjusted_price = current_bid - slippage
            else:  # Short position
                adjusted_price = current_ask + slippage

            # Calculate PnL
            pnl = ((adjusted_price - open_price) * direction - commission_per_lot) * volume
            positions[i]['pnl'] = pnl

            if closing_method == 1:
                # Standard stop-loss logic
                if pnl >= profit_threshold or pnl <= -loss_threshold:
                    positions[i]['close_price'] = adjusted_price
                    positions[i]['close_id'] = tick_id
                    positions[i]['pnl'] = pnl
            elif closing_method == 2:
                # Trailing stop logic
                # trailing_stop = open_price * (1 + (profit_threshold if direction == 1 else -profit_threshold))
                # if (direction == 1 and adjusted_price < trailing_stop) or (direction == -1 and adjusted_price > trailing_stop):
                #     positions[i]['close_price'] = adjusted_price
                #     positions[i]['close_id'] = tick_id
                #     positions[i]['pnl'] = pnl
                if direction == 1:  # Long position
                    highest_price = max(positions[i]['highest_price'], adjusted_price)
                    positions[i]['highest_price'] = highest_price
                    trailing_stop = highest_price * (1 - trailing_pct)
                    if adjusted_price < trailing_stop:
                        positions[i]['close_price'] = adjusted_price
                        positions[i]['close_id'] = tick_id
                        positions[i]['pnl'] = pnl
                else:  # Short position
                    lowest_price = min(positions[i]['lowest_price'], adjusted_price)
                    positions[i]['lowest_price'] = lowest_price
                    trailing_stop = lowest_price * (1 + trailing_pct)
                    if adjusted_price > trailing_stop:
                        positions[i]['close_price'] = adjusted_price
                        positions[i]['close_id'] = tick_id
                        positions[i]['pnl'] = pnl
            else:
                raise ValueError(f"Invalid closing_method: {closing_method}")
            
    # return positions
            
# @njit
def calculate_pnl_numba(open_prices, close_prices, directions, volumes, commissions):
    """
    Calculate PnL for multiple positions using Numba.

    Parameters:
    - open_prices (np.ndarray): Array of open prices.
    - close_prices (np.ndarray): Array of close prices.
    - directions (np.ndarray): Array of directions (1 or -1).
    - volumes (np.ndarray): Array of volumes.
    - commissions (np.ndarray): Array of commissions.

    Returns:
    - np.ndarray: Array of PnL values.
    """
    pnls = np.empty(len(open_prices), dtype=np.float32)
    for i in range(len(open_prices)):
        pnl = ((close_prices[i] - open_prices[i]) * directions[i]  - commissions[i])* volumes[i]
        pnls[i] = pnl
    return pnls

# @njit(parallel=True)
def calculate_floating_pnl_numba(positions, current_bid, current_ask, slippage, commission):
    """
    Calculate the total floating (unrealized) PnL for all open positions.

    Parameters:
    - positions (np.ndarray): Array of positions.
    - current_bid (float): Current bid price.
    - current_ask (float): Current ask price.
    - slippage (float): Slippage applied to the price.

    Returns:
    - float: Total floating PnL.
    """
    total_pnl = 0.0
    for pos in positions:
        if pos['close_id'] == -1:
            direction = pos['direction']
            # direction *= -1
            open_price = pos['open_price']
            volume = pos['volume']
            
            if direction == 1:  # Long position
                current_price = current_bid - slippage
            else:  # Short position
                current_price = current_ask + slippage
            pnl = ((current_price - open_price) * direction - commission) * volume
            total_pnl += pnl
    return total_pnl