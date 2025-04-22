from functools import wraps


class ValidationFailedException(Exception):
    """Raised when the extraction result does not meet validation requirements."""

    pass


def validate_output(validator):
    """
    Decorator to validate the output of an extraction method.

    Args:
        validator (callable): A function that takes the result as input and returns True if valid.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
            except Exception as ex:
                raise ex  # Propagate the exception so fallback can be triggered.
            if not validator(result):
                raise ValidationFailedException(
                    f"Validation failed for output: {result}"
                )
            return result

        return wrapper

    return decorator