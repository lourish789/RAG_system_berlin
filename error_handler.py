"""
Error handling utilities with graceful degradation
"""
import logging
import functools
from flask import jsonify

logger = logging.getLogger(__name__)


class ArchiveError(Exception):
    """Base exception for archive errors"""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def handle_errors(f):
    """
    Decorator for graceful error handling in Flask routes
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ArchiveError as e:
            logger.error(f"Archive error in {f.__name__}: {e.message}")
            return jsonify({
                'error': e.message,
                'type': 'ArchiveError'
            }), e.status_code
        except FileNotFoundError as e:
            logger.error(f"File not found in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'File not found or corrupted',
                'type': 'FileNotFoundError',
                'details': 'Please ensure the file exists and is not corrupted'
            }), 404
        except TimeoutError as e:
            logger.error(f"Timeout in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Operation timed out',
                'type': 'TimeoutError',
                'details': 'The processing service may be unavailable. Please try again.'
            }), 504
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'An unexpected error occurred',
                'type': type(e).__name__,
                'details': str(e)
            }), 500
    
    return decorated_function