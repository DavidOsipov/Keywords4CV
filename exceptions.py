"""Custom exceptions for the Keywords4CV application."""


class ConfigError(Exception):
    """Raised when there is an error in the configuration."""

    pass


class InputValidationError(Exception):
    """Raised when there is an error in the input data."""

    pass


class DataIntegrityError(Exception):
    """Raised when there is an error in the data integrity."""

    pass


class APIError(Exception):
    """Raised when there is an error in API communication."""

    pass


class NetworkError(Exception):
    """Raised when there is a network error."""

    pass


class AuthenticationError(Exception):
    """Raised when there is an authentication error."""

    pass


class SimpleCircuitBreaker:
    """A simple circuit breaker implementation."""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def __call__(self, func):
        """Decorator to wrap a function with circuit breaker logic."""

        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                # Check if recovery timeout has elapsed
                if (
                    self.last_failure_time
                    and time.time() - self.last_failure_time > self.recovery_timeout
                ):
                    self.state = "HALF-OPEN"
                else:
                    raise NetworkError("Circuit breaker is open")

            try:
                result = func(*args, **kwargs)
                if self.state == "HALF-OPEN":
                    # Success, close the circuit
                    self.failure_count = 0
                    self.state = "CLOSED"
                return result
            except (APIError, NetworkError) as e:
                # Only count API and Network errors
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                raise e
            except Exception as e:
                # Don't count other exceptions towards circuit breaker
                raise e

        return wrapper
