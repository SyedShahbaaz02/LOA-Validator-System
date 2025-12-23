"""
Aggressive Retry Decorator for GPT-4o API Calls
Ensures GPT-4o vision succeeds by retrying with exponential backoff
NO FALLBACK - Either succeeds or returns ERROR
"""

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _calculate_retry_delay(
    error_str: str,
    current_delay: float,
    max_delay: float,
    backoff_multiplier: float,
    log_message: bool = True,
) -> float:
    """
    Calculate the next delay value based on error type.

    This function:
    1. Logs the CURRENT delay (what we're sleeping with NOW)
    2. Calculates and returns the NEXT delay (for the next iteration)

    Args:
        error_str: The error message/string to analyze
        current_delay: Current delay value to sleep with NOW
        max_delay: Maximum allowed delay
        backoff_multiplier: Multiplier for exponential backoff
        log_message: Whether to log the delay message (default: True)

    Returns:
        float: The NEXT delay value for the next iteration
    """
    is_429_error = "429" in error_str or "rate" in error_str.lower()
    is_500_error = "500" in error_str or "internal" in error_str.lower()

    if is_429_error:
        # More aggressive backoff for rate limits (3x multiplier)
        next_delay = min(current_delay * 3, max_delay)
        if log_message:
            logger.info(
                f"Rate limit detected - waiting {current_delay:.1f} seconds before retry..."
            )
    elif is_500_error:
        # Standard exponential backoff for server errors
        next_delay = min(current_delay * backoff_multiplier, max_delay)
        if log_message:
            logger.info(
                f"Server error detected - waiting {current_delay:.1f} seconds before retry..."
            )
    else:
        # Standard exponential backoff for other errors
        next_delay = min(current_delay * backoff_multiplier, max_delay)
        if log_message:
            logger.info(f"Waiting {current_delay:.1f} seconds before retry...")

    return next_delay


def aggressive_retry(
    max_attempts: int = 10,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
):
    """
    Decorator that implements aggressive retry logic for API calls (especially GPT-4o vision).

    Args:
        max_attempts: Maximum number of retry attempts (default: 10)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)

    Retry Strategy:
    - 500 errors (Internal Server Error): Retry aggressively - temporary service issue
    - 429 errors (Rate Limit): Retry with longer delays - rate limiting
    - 400 errors (Bad Request): Do NOT retry - permanent error in request
    - Other errors: Retry with standard backoff

    The decorator will keep retrying until:
    1. Request succeeds (returns result)
    2. Max attempts reached (raises final exception)
    3. Receives 400 error (raises immediately - no point retrying bad request)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    # Attempt the function call
                    logger.info(
                        f"GPT-4o Vision Call Attempt {attempt}/{max_attempts} for {func.__name__}"
                    )
                    result = func(*args, **kwargs)

                    # Check if result indicates failure (for functions that return dicts with 'success' key)
                    if (
                        isinstance(result, dict)
                        and "success" in result
                        and not result["success"]
                    ):
                        error_msg = result.get("error", "Unknown error")

                        # Check if error is a 400 (bad request) - don't retry, return ERROR status
                        if (
                            "400" in error_msg
                            or "invalid_type" in error_msg.lower()
                            or "invalid_request" in error_msg.lower()
                        ):
                            logger.error(
                                f"GPT-4o Vision returned 400 Bad Request - NOT retrying: {error_msg}"
                            )
                            return {
                                "success": False,
                                "error": f"GPT-4o Vision Bad Request (400): {error_msg}",
                                "extraction_log": kwargs.get("extraction_log", {}),
                                "error_type": "bad_request_400",
                            }

                        # For other errors (500, 429, etc.), retry
                        logger.warning(f"GPT-4o Vision returned error: {error_msg}")
                        last_exception = Exception(error_msg)

                        if attempt < max_attempts:
                            # Sleep with current delay, then calculate next delay for next iteration
                            time.sleep(delay)
                            delay = _calculate_retry_delay(
                                error_msg, delay, max_delay, backoff_multiplier
                            )
                        continue

                    # Success!
                    if attempt > 1:
                        logger.info(f"✓ GPT-4o Vision succeeded on attempt {attempt}")
                    return result

                except Exception as e:
                    last_exception = e
                    error_str = str(e)

                    # Check error type
                    is_400_error = (
                        "400" in error_str
                        or "invalid_type" in error_str.lower()
                        or "invalid_request" in error_str.lower()
                    )

                    if is_400_error:
                        # Bad request - don't retry, return ERROR status
                        logger.error(
                            f"GPT-4o Vision 400 Bad Request - NOT retrying: {error_str}"
                        )
                        return {
                            "success": False,
                            "error": f"GPT-4o Vision Bad Request (400): {error_str}",
                            "extraction_log": kwargs.get("extraction_log", {}),
                            "error_type": "bad_request_400",
                        }

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {error_str}"
                    )

                    if attempt < max_attempts:
                        # Sleep with current delay, then calculate next delay for next iteration
                        time.sleep(delay)
                        delay = _calculate_retry_delay(
                            error_str, delay, max_delay, backoff_multiplier
                        )
                    else:
                        logger.error(
                            f"✗ GPT-4o Vision FAILED after {max_attempts} attempts"
                        )

            # If we get here, all attempts failed - return ERROR status to be handled by caller
            error_msg = f"GPT-4o Vision failed after {max_attempts} attempts. Last error: {last_exception}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "extraction_log": kwargs.get("extraction_log", {}),
                "error_type": "max_attempts_exceeded",
            }

        return wrapper

    return decorator
