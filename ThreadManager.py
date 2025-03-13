# thread_manager.py

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event

class ThreadManager:
    def __init__(self, max_workers=4):
        """
        Initialize the ThreadManager with a ThreadPoolExecutor.
        
        Parameters:
        - max_workers (int): Maximum number of threads in the pool.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("ThreadManager initialized with ThreadPoolExecutor.")

    def execute_concurrently(self, tasks):
        """
        Execute a list of tasks concurrently.
        
        Parameters:
        - tasks (list): List of callable tasks.
        
        Returns:
        - list: Results from the executed tasks.
        """
        futures = [self.executor.submit(task) for task in tasks]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error executing task: {e}")
        return results

    def shutdown(self, wait=True):
        """
        Shutdown the ThreadPoolExecutor.
        
        Parameters:
        - wait (bool): If True, wait for all threads to finish.
        """
        self.logger.debug("ThreadManager initiating shutdown.")
        self.executor.shutdown(wait=wait)
        self.logger.debug("ThreadManager shutdown complete.")