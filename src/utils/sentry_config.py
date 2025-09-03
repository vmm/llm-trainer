"""Configuration utilities for Sentry error reporting.

This module provides utilities to properly configure or disable Sentry SDK
to prevent "No valid Sentry data received" errors when Sentry is not intended
to be used but is present as a dependency (e.g., from WandB).
"""

import os
import logging
from typing import Optional


def disable_sentry() -> None:
    """
    Disable Sentry SDK to prevent error reporting attempts.
    
    This function should be called early in the application lifecycle to prevent
    Sentry from attempting to send error data when not properly configured.
    """
    try:
        import sentry_sdk
        
        # Initialize Sentry with disabled transport to prevent any data sending
        sentry_sdk.init(
            dsn=None,  # No DSN means no data will be sent
            traces_sample_rate=0.0,  # Disable performance monitoring
            profiles_sample_rate=0.0,  # Disable profiling
            send_default_pii=False,  # Don't send personally identifiable information
            debug=False,  # Disable debug mode
            environment="disabled",  # Mark environment as disabled
        )
        
        # Set environment variable to ensure WandB doesn't try to use Sentry
        os.environ["WANDB_DISABLE_SENTRY"] = "true"
        
        logging.debug("Sentry SDK has been disabled to prevent error reporting")
        
    except ImportError:
        # Sentry SDK not installed, nothing to disable
        logging.debug("Sentry SDK not found, no action needed")
    except Exception as e:
        # Log any issues but don't fail the application
        logging.warning(f"Could not disable Sentry SDK: {e}")


def configure_sentry(dsn: Optional[str] = None, environment: str = "production") -> None:
    """
    Configure Sentry SDK with proper settings.
    
    Args:
        dsn: Sentry Data Source Name. If None, Sentry will be disabled.
        environment: Environment name for Sentry reporting.
    """
    try:
        import sentry_sdk
        
        if dsn is None:
            # No DSN provided, disable Sentry
            disable_sentry()
            return
        
        # Configure Sentry with provided DSN
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=0.1,  # Capture 10% of transactions for performance monitoring
            profiles_sample_rate=0.1,  # Capture 10% of profiles
            send_default_pii=False,  # Don't send personally identifiable information
            environment=environment,
            before_send=_filter_sentry_events,
        )
        
        logging.info(f"Sentry SDK configured for environment: {environment}")
        
    except ImportError:
        logging.debug("Sentry SDK not installed")
    except Exception as e:
        logging.error(f"Failed to configure Sentry SDK: {e}")
        # Fallback to disabling Sentry
        disable_sentry()


def _filter_sentry_events(event, hint):
    """
    Filter Sentry events to avoid sending sensitive or unnecessary data.
    
    Args:
        event: The Sentry event dictionary
        hint: Additional context about the event
        
    Returns:
        The filtered event or None to drop it
    """
    # Don't send events for common/expected errors that aren't actionable
    if "exception" in event:
        for exception in event["exception"]["values"]:
            exc_type = exception.get("type", "")
            exc_value = exception.get("value", "")
            
            # Filter out common package/dependency warnings
            if any(term in exc_value.lower() for term in [
                "deprecationwarning", 
                "futurewarning",
                "userwarning",
                "no valid sentry data",
            ]):
                return None
    
    return event