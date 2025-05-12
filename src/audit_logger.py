import logging
import os
from pathlib import Path

# Define absolute path within the container explicitly
APP_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = APP_DIR / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "audit.log"

# Configure the audit logger
# This logger will be used specifically for audit trails.

logger = logging.getLogger("audit")
logger.setLevel(logging.INFO) # Capture info and higher level messages

# Define dummy extra fields for setup logs
SETUP_EXTRA = {
    "user": "SYSTEM_SETUP", 
    "role": "N/A", 
    "action": "LOGGER_SETUP", 
    "model": "N/A", 
    "links": "N/A", 
    "details": "Logger setup process"
}

# Explicitly create the directory *before* handler setup
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured logs directory exists: {LOGS_DIR}", extra=SETUP_EXTRA)
except Exception as e:
    logger.error(f"Failed to create logs directory {LOGS_DIR}: {e}", extra=SETUP_EXTRA)

# --- File Handler Setup with Error Handling ---
file_handler = None
try:
    file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding='utf-8')
    # Define formatter here
    formatter = logging.Formatter(
        "%(asctime)s | USER: %(user)s | ROLE: %(role)s | ACTION: %(action)s | MODEL: %(model)s | LINKS: %(links)s | DETAILS: %(details)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.info(f"FileHandler configured for {AUDIT_LOG_FILE}", extra=SETUP_EXTRA) # Console log confirmation
except Exception as e:
    logger.error(f"Failed to initialize FileHandler for {AUDIT_LOG_FILE}: {e}", extra=SETUP_EXTRA)
    # file_handler remains None

# --- Stream Handler Setup --- 
stream_handler = logging.StreamHandler() # Defaults to stderr
# Ensure formatter is defined even if file_handler failed, for stream_handler
if 'formatter' not in locals():
     formatter = logging.Formatter(
        "%(asctime)s | USER: %(user)s | ROLE: %(role)s | ACTION: %(action)s | MODEL: %(model)s | LINKS: %(links)s | DETAILS: %(details)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
stream_handler.setFormatter(formatter) # Use the same format for console output
logger.info("StreamHandler configured.", extra=SETUP_EXTRA) # Console log confirmation

# Add the handlers to the logger if they haven't been added yet
# Check prevents duplicate handlers if this module is reloaded
if not logger.handlers:
    if file_handler: # Only add if successfully initialized
        logger.addHandler(file_handler)
        logger.info("FileHandler added to logger.", extra=SETUP_EXTRA)
    else:
        # Use logger.warning for non-critical setup issues
        logger.warning("FileHandler was not initialized, skipping addHandler.", extra=SETUP_EXTRA)
    
    # Check if stream handler already added (less likely to cause issues but good practice)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(stream_handler)
        logger.info("StreamHandler added to logger.", extra=SETUP_EXTRA)

# Helper function to log audit events
def log_audit_event(username: str, 
                    role: str, 
                    action: str, 
                    details: str, 
                    links: list[str] | None = None, 
                    model_name: str | None = None): # Added model_name parameter
    """
    Logs an audit event with a specific format.
    Uses extra context for the logger.
    """
    processed_links = ", ".join(links) if links and isinstance(links, list) else "N/A"
    model_for_log = model_name if model_name else "N/A"
    
    logger.info(
        details, # The main message for logger.info
        extra={"user": username if username else "SYSTEM", 
               "role": role if role else "N/A",
               "action": action,
               "model": model_for_log, # Pass model name to formatter
               "links": processed_links,
               "details": details # Passing details again for the custom formatter field
              }
    )

# Example of how to use it (for testing this module directly):
if __name__ == "__main__":
    # These direct calls won't have the 'user' and 'action' in the formatter unless passed in extra
    # The log_audit_event function is the intended way to use this logger.
    log_audit_event("test_user", "admin", "TEST_ACTION", "This is a test log event.")
    log_audit_event("another_user", "researcher", "ANOTHER_TEST", "Another test detail.")
    log_audit_event("link_submitter", "researcher", "LINKS_SUBMITTED", "User provided web links for research.", links=["http://example.com", "https://anotherexample.org"])
    log_audit_event("no_link_user", "editor", "CONTENT_EDIT", "User edited content without providing links.")
    print(f"Audit log should be in: {AUDIT_LOG_FILE}") 