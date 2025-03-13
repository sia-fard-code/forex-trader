# PositionClosureHandler.py

import logging

class PositionClosureHandler:
    def __init__(self, trade_manager, config):
        """
        Initialize the PositionClosureHandler.
        
        Parameters:
        - trade_manager (TradeManager): Instance managing trades.
        - config (dict): Configuration parameters.
        """
        self.trade_manager = trade_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def handle_closures(self, refined_state, tick_id, bid, ask):
        """
        Identify and close positions based on PnL thresholds.
        
        Parameters:
        - refined_state (int): Current market state.
        - tick_id (int): Current tick identifier.
        - bid (float): Current bid price.
        - ask (float): Current ask price.
        """
        # Identify positions to close
        positions_to_close = []
        positions_to_close = self.trade_manager.identify_positions_to_close(
            profit_threshold=self.config.get("profit_threshold", 50.0),
            loss_threshold=self.config.get("loss_threshold", 20.0)
        )
        for position in positions_to_close:
            # Calculate close price with slippage
            close_price = self.calculate_close_price(position, bid, ask)
            
            # Execute closure
            self.trade_manager.close_position_by_id(
                position_id=position['id'],
                close_price=close_price,
                pnl=position['pnl'],
                tick_id=tick_id
            )
            self.logger.debug(f"Closed position {position['id']} at tick_id={tick_id}")
    
    def calculate_close_price(self, position, bid, ask):
        """
        Calculate the close price with slippage based on position direction.
        
        Parameters:
        - position (dict): Position details.
        - bid (float): Current bid price.
        - ask (float): Current ask price.
        
        Returns:
        - float: Calculated close price.
        """
        if position['direction'] == 1:  # Long
            return bid - position['slippage']
        else:  # Short
            return ask + position['slippage']