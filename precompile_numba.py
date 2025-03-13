from numba_functions import update_positions_numba_parallel, calculate_pnl_numba
import numpy as np

def precompile_numba_functions():
    # Dummy data for precompilation
    positions = np.empty(1, dtype=[
        ('close_id', 'i8'), ('direction', 'i1'), ('open_price', 'f4'),
        ('volume', 'f4'), ('pnl', 'f4'), ('close_price', 'f4')
    ])
    current_bid, current_ask = 1.2, 1.3
    t, profit_threshold, loss_threshold = 0, 0.005, 0.002
    slippage, commission_per_lot = 0.0001, 2.0

    # Precompile functions
    update_positions_numba_parallel(positions, current_bid, current_ask, t, profit_threshold, loss_threshold, slippage, commission_per_lot)

    open_prices = np.array([1.2], dtype=np.float32)
    close_prices = np.array([1.3], dtype=np.float32)
    directions = np.array([1], dtype=np.int8)
    volumes = np.array([0.1], dtype=np.float32)
    commissions = np.array([2.0], dtype=np.float32)

    calculate_pnl_numba(open_prices, close_prices, directions, volumes, commissions)

if __name__ == "__main__":
    precompile_numba_functions()