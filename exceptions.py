"""
Custom exceptions for the Keywords4CV application.

This module defines custom exceptions used throughout the application to
provide more specific error handling and reporting.
"""


class Keywords4CVError(Exception):
    """Base exception for all keywords4cv errors."""


class ConfigError(Keywords4CVError):
    """Raised when there's a configuration error."""


class InputValidationError(Keywords4CVError):
    """Raised when input validation fails."""


class CriticalFailureError(Keywords4CVError):
    """Raised when a critical failure occurs."""


class AggregationError(Keywords4CVError):
    """Raised when result aggregation fails."""


class DataIntegrityError(Keywords4CVError):
    """Raised when data integrity checks fail."""
