"""
Utility functions and classes for AWEsomeScoring.

This module contains common utilities used across the application.
"""

import logging
import random
import time
from typing import Any, Callable, Dict, List, Tuple, Union, Optional

logger = logging.getLogger("awesome_scoring")

class RetryHandler:
    """
    Handles retrying API calls with exponential backoff.
    
    This class provides utility methods for retrying API calls with
    exponential backoff and jitter to handle rate limits and transient errors.
    """
    
    @staticmethod
    def retry_with_backoff(
        func: Callable[[], Any], 
        max_retries: int = 12, 
        initial_wait: float = 2.0,
        max_wait: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: float = 0.1,
        retry_on: Tuple = (Exception,),
        description: str = "API Call"
    ) -> Any:
        """
        Execute a function with exponential backoff retry strategy.
        
        Args:
            func: The function to execute
            max_retries: Maximum number of retry attempts
            initial_wait: Initial wait time in seconds
            max_wait: Maximum wait time in seconds
            backoff_factor: Factor to multiply wait time by after each failure
            jitter: Random jitter factor to avoid thundering herd problem
            retry_on: Tuple of exception types to retry on
            description: Description of the operation for logging
            
        Returns:
            The result of the function call
            
        Raises:
            Exception: If all retry attempts fail
        """
        wait_time = initial_wait
        
        for attempt in range(max_retries):
            try:
                return func()
                
            except retry_on as e:
                ## Check if it's the last attempt
                if attempt == max_retries - 1:
                    logger.error(f"{description} failed after {max_retries} attempts: {str(e)}")
                    raise
                
                ## Add random jitter to avoid thundering herd problem
                jitter_amount = random.uniform(-jitter * wait_time, jitter * wait_time)
                adjusted_wait_time = wait_time + jitter_amount
                
                logger.warning(
                    f"{description} attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                    f"Retrying in {adjusted_wait_time:.2f} seconds..."
                )
                
                time.sleep(max(0.1, adjusted_wait_time))  ## Ensure wait time is positive
                
                ## Increase wait time exponentially, capped at max_wait
                wait_time = min(wait_time * backoff_factor, max_wait)
        
        ## This should not be reached due to the raise in the loop
        raise Exception(f"Maximum number of retries ({max_retries}) reached. Aborting.")


def format_time_elapsed(seconds: float) -> str:
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string (e.g., "2h 30m 15s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def truncate_text(text: str, max_length: int = 100, ellipsis: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length of the output text
        ellipsis: String to append if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(ellipsis)] + ellipsis


def clean_filename(filename: str) -> str:
    """
    Clean a string to be used as a filename.
    
    Args:
        filename: Input filename
        
    Returns:
        Cleaned filename
    """
    import re
    ## Replace any non-alphanumeric characters with underscores
    clean = re.sub(r'[^\w\-\.]', '_', filename)
    ## Remove multiple consecutive underscores
    clean = re.sub(r'_+', '_', clean)
    return clean
