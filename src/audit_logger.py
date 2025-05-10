import logging
import os
from pathlib import Path

# Ensure logs directory exists (assuming config.py or main.py creates it, 
# but good to have a check here too if this module is used independently)
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_FILE = LOGS_DIR / "audit.log"

# Configure the audit logger
# This logger will be used specifically for audit trails.

logger = logging.getLogger("audit")
logger.setLevel(logging.INFO) # Capture info and higher level messages

# Create a file handler for the audit log
# Use a rotating file handler if logs are expected to be large over time
# For now, a simple FileHandler will suffice.
file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding='utf-8')

# Define the log format
# Example: 2023-10-27 10:30:00,123 | USER: admin | ROLE: admin | ACTION: LOGIN | DETAILS: Successful login
# New Example: 2023-10-27 10:30:00,123 | USER: admin | ROLE: admin | ACTION: WEB_RESEARCH | LINKS: http://a.com, http://b.com | DETAILS: Submitted research
formatter = logging.Formatter("%(asctime)s | USER: %(user)s | ROLE: %(role)s | ACTION: %(action)s | LINKS: %(links)s | DETAILS: %(details)s", 
                              datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)

# Add the handler to the logger
if not logger.handlers:
    logger.addHandler(file_handler)

# Helper function to log audit events
def log_audit_event(username: str, role: str, action: str, details: str, links: list[str] | None = None):
    """
    Logs an audit event with a specific format.
    Uses extra context for the logger.
    """
    processed_links = ", ".join(links) if links and isinstance(links, list) else "N/A"
    
    logger.info(
        details, # The main message for logger.info
        extra={"user": username if username else "SYSTEM", 
               "role": role if role else "N/A",
               "action": action,
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