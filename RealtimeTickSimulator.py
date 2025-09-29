import logging
import pandas as pd


class RealtimeTickSimulator:
    def __init__(self, strategy_ref, max_processing_time_ms=20):
        self.strategy = strategy_ref
        self.max_processing_time = max_processing_time_ms / 1000.0
        self.last_tick_timestamp = None
        self.processing_finish_time = None
        self.dropped_ticks = 0
        self.processed_ticks = 0
        self.training_ticks = 0
        
    def should_process_tick(self, tick_timestamp, estimated_processing_time=None):
        """
        Process all ticks during training phase, apply filtering only when trained
        """
        # Check if we're still in training phase
        is_trained = self.strategy.data_manager.get_config("is_trained")
        
        if not is_trained:
            # During training, process ALL ticks - no filtering
            self.training_ticks += 1
            self.last_tick_timestamp = tick_timestamp
            return True
            
        # Only apply realistic filtering AFTER training is complete
        return self._apply_realistic_filtering(tick_timestamp, estimated_processing_time)
    
    def _apply_realistic_filtering(self, tick_timestamp, estimated_processing_time):
        """Apply realistic processing constraints only after training"""
        
        if estimated_processing_time is None:
            stats = self.strategy.get_processing_time_stats()
            estimated_processing_time = stats.get('average', 0.005) * 1.2
            
        if self.last_tick_timestamp is None:
            self.last_tick_timestamp = tick_timestamp
            self.processing_finish_time = tick_timestamp + pd.Timedelta(seconds=estimated_processing_time)
            self.processed_ticks += 1
            return True
            
        time_between_ticks = (tick_timestamp - self.last_tick_timestamp).total_seconds()
        
        # Check if we're still processing previous tick
        if self.processing_finish_time and tick_timestamp < self.processing_finish_time:
            self.dropped_ticks += 1
            logging.debug(f"Dropped tick at {tick_timestamp} - still processing previous")
            return False
            
        # Check if processing time would exceed available time
        if estimated_processing_time > time_between_ticks * 0.8:
            self.dropped_ticks += 1
            logging.debug(f"Dropped tick - processing too slow ({estimated_processing_time:.3f}s > {time_between_ticks:.3f}s)")
            return False
            
        self.last_tick_timestamp = tick_timestamp
        self.processing_finish_time = tick_timestamp + pd.Timedelta(seconds=estimated_processing_time)
        self.processed_ticks += 1
        return True
    
    def get_stats(self):
        return {
            'training_ticks': self.training_ticks,
            'processed_ticks': self.processed_ticks,
            'dropped_ticks': self.dropped_ticks,
            'total_post_training': self.processed_ticks + self.dropped_ticks,
            'processing_efficiency': self.processed_ticks / (self.processed_ticks + self.dropped_ticks) if (self.processed_ticks + self.dropped_ticks) > 0 else 1.0
        }