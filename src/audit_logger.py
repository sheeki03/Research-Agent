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
logger.propagate = False # Prevent propagation to parent/root handlers

# Define a simple formatter for setup messages
setup_formatter = logging.Formatter('%(asctime)s | SETUP | %(levelname)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

# Temporarily add a stream handler with simple format for setup diagnostics
setup_stream_handler = logging.StreamHandler()
setup_stream_handler.setFormatter(setup_formatter)
logger.addHandler(setup_stream_handler)

# --- Perform Setup Logging (using the temporary handler) ---
logger.info(f"Initializing audit logger.")
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured logs directory exists: {LOGS_DIR}")
except Exception as e:
    logger.error(f"Failed to create logs directory {LOGS_DIR}: {e}")

# --- Configure Final Handlers --- 

# Define the detailed formatter for actual audit events
audit_formatter = logging.Formatter(
    "%(asctime)s | USER: %(user)s | ROLE: %(role)s | ACTION: %(action)s | MODEL: %(model)s | LINKS: %(links)s | DETAILS: %(details)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- File Handler Setup with Error Handling ---
file_handler = None
try:
    file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(audit_formatter) # Use detailed formatter
    logger.info(f"FileHandler configured for {AUDIT_LOG_FILE}")
except Exception as e:
    logger.error(f"Failed to initialize FileHandler for {AUDIT_LOG_FILE}: {e}")
    file_handler = None

# --- Stream Handler Setup --- 
stream_handler = logging.StreamHandler() 
stream_handler.setFormatter(audit_formatter) # Use detailed formatter
logger.info("Final StreamHandler configured with detailed format.")

# --- Finalize Logger Configuration ---

# Remove the temporary setup handler now that setup logging is done
logger.removeHandler(setup_stream_handler)
logger.info("Removed temporary setup handler.")

# Clear any OTHER handlers that might have been attached (e.g., inherited/duplicates)
# Important to do this AFTER setup logging and BEFORE adding final handlers
for handler in logger.handlers[:]: # Iterate over a copy
    logger.removeHandler(handler)
logger.info(f"Cleared existing handlers (if any). Current handlers: {logger.handlers}")

# Add the final handlers 
if file_handler:
    logger.addHandler(file_handler)
    logger.info("Final FileHandler added.")
else:
    logger.warning("Final FileHandler was not initialized or added.")

logger.addHandler(stream_handler)
logger.info("Final StreamHandler added.")
logger.info(f"Audit logger initialization complete. Final handlers: {logger.handlers}")

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