import streamlit as st
import yaml
import bcrypt
import os
from pathlib import Path
import asyncio
from typing import List, Dict, Any, Optional
import io

try:
    import fitz  # PyMuPDF
except ImportError:
    # This will be caught at runtime if not installed. 
    # User will need to add PyMuPDF to requirements.txt and rebuild.
    pass 
try:
    from docx import Document # python-docx
except ImportError:
    # This will be caught at runtime if not installed.
    # User will need to add python-docx to requirements.txt and rebuild.
    pass

from src.config import (
    DEBUG,
    LOG_LEVEL,
    OUTPUT_FORMAT,
    OUTPUT_DIR,
    DEFAULT_PROMPTS,
    USERS_CONFIG_PATH,
    SYSTEM_PROMPT,
    OPENROUTER_PRIMARY_MODEL # Import the primary model default
)
from src.openrouter import OpenRouterClient
from src.firecrawl_client import FirecrawlClient
from src.audit_logger import log_audit_event # Added for Task 5

# Initialize clients
@st.cache_resource
def init_clients():
    openrouter_client = OpenRouterClient()
    firecrawl_client = FirecrawlClient(
        redis_url=os.getenv("REDIS_URL")
    )
    return openrouter_client, firecrawl_client

def load_users() -> Dict[str, Any]:
    """Load user data from YAML file."""
    if not os.path.exists(USERS_CONFIG_PATH):
        # If no users.yaml, create it with default admin/researcher from init_users.py logic
        # This is a fallback, ideally init_users.py should be run once.
        from src.init_users import init_users as initialize_system_users
        try:
            initialize_system_users()
            st.info("User configuration file not found. Initialized with default users.")
        except Exception as e:
            st.error(f"Failed to initialize default users: {e}")
            return {}
            
    with open(USERS_CONFIG_PATH, 'r') as f:
        users = yaml.safe_load(f)
        if users is None: # Handle empty users.yaml file
            st.warning("User configuration file is empty. Please run user initialization or sign up.")
            return {}
        return users

def save_users(users_data: Dict[str, Any]) -> bool:
    try:
        with open(USERS_CONFIG_PATH, 'w') as f:
            yaml.dump(users_data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Failed to save user data: {e}")
        return False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(
        plain_password.encode(),
        hashed_password.encode()
    )

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# Helper functions for document processing
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF file bytes."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text from DOCX file bytes."""
    try:
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return ""

def extract_text_from_txt_md(file_bytes: bytes, encoding='utf-8') -> str:
    """Extracts text from TXT or MD file bytes."""
    try:
        return file_bytes.decode(encoding)
    except UnicodeDecodeError:
        try:
            return file_bytes.decode('latin-1') # Fallback encoding
        except Exception as e:
            st.error(f"Error decoding text file with fallback: {e}")
            return ""
    except Exception as e:
        st.error(f"Error processing text/markdown file: {e}")
        return ""
# End of helper functions

async def process_urls(urls: List[str], client: FirecrawlClient) -> List[Dict[str, str]]:
    """Process a list of URLs and return a list of dicts with url, content/error."""
    results = await client.scrape_multiple_urls(urls)
    
    processed_results = []
    for result in results:
        url = result.get("metadata", {}).get("url", result.get("url", "unknown URL")) # Get URL from metadata or direct key
        if result.get("success", False):
            content_data = result.get("data", {}).get("content", "")
            if not content_data: # Fallback if 'data' structure is not present
                 content_data = result.get("content", "")
            processed_results.append({"url": url, "content": content_data, "status": "success"})
        else:
            error_message = result.get("error", "Unknown error")
            processed_results.append({"url": url, "error": error_message, "status": "failed"})
            # UI warning will be handled where this function is called, based on the status
    
    return processed_results

async def perform_web_research(query: str, client: OpenRouterClient) -> str:
    """Perform web research on a given query."""
    # Generate search queries
    queries = await client.generate_serp_queries(query)
    
    # Simulate search results (replace with actual search implementation)
    search_results = {
        "data": [
            {"content": "Sample content 1", "url": "https://example.com/1"},
            {"content": "Sample content 2", "url": "https://example.com/2"}
        ]
    }
    
    # Process search results
    results = await client.process_serp_result(query, search_results)
    
    # Write final report
    report = await client.write_final_report(
        query,
        results["learnings"],
        [item["url"] for item in search_results["data"]]
    )
    
    return report

# --- Add Model Definitions Here ---
# Define model choices based on user query and web search results
MODEL_OPTIONS = {
    "Mistral Medium 3": "mistralai/mistral-medium-3",
    "Google Gemini 2.5 Pro": "google/gemini-2.5-pro-preview",
    "OpenAI o3": "openai/o3",
    "OpenAI GPT-4.1": "openai/gpt-4.1",
    "Claude 3.7 Sonnet (thinking)": "anthropic/claude-3.7-sonnet:thinking",
    "DeepSeek R1T Chimera": "tngtech/deepseek-r1t-chimera:free",
}
MODEL_DISPLAY_NAMES = list(MODEL_OPTIONS.keys())
MODEL_IDENTIFIERS = list(MODEL_OPTIONS.values())
# --- End Model Definitions ---

async def main():
    st.set_page_config(
        page_title="AI Research Agent",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state for system_prompt if not already set
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPT # Global default
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "show_signup" not in st.session_state:
        st.session_state.show_signup = False
    if "processed_documents_content" not in st.session_state: 
        st.session_state.processed_documents_content = []
    if "last_uploaded_file_details" not in st.session_state: # For comparing if files changed
        st.session_state.last_uploaded_file_details = []
    if "unified_report_content" not in st.session_state: 
        st.session_state.unified_report_content = ""
    if "scraped_web_content" not in st.session_state: # Added for this task (Task 3)
        st.session_state.scraped_web_content = []

    # Sidebar
    with st.sidebar:
        st.title("Access Panel")
        
        if not st.session_state.authenticated:
            if st.session_state.show_signup:
                st.subheader("Create Account")
                new_username = st.text_input("New Username", key="signup_username")
                new_password = st.text_input("New Password", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
                
                if st.button("Sign Up", key="signup_submit_button"):
                    if not new_username or not new_password:
                        st.error("Username and password cannot be empty.")
                        log_audit_event(new_username or "ANONYMOUS", "N/A", "USER_SIGNUP_FAILURE", "Attempted signup with empty username/password.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                        log_audit_event(new_username, "N/A", "USER_SIGNUP_FAILURE", "Attempted signup with non-matching passwords.")
                    else:
                        users = load_users()
                        if new_username in users:
                            st.error("Username already exists.")
                            log_audit_event(new_username, "N/A", "USER_SIGNUP_FAILURE", f"Attempted signup with existing username: {new_username}.")
                        else:
                            users[new_username] = {
                                "password": hash_password(new_password),
                                "role": "researcher", # Default role
                                "system_prompt": DEFAULT_PROMPTS.get("researcher", SYSTEM_PROMPT)
                            }
                            if save_users(users):
                                st.success("Account created successfully! Please log in.")
                                log_audit_event(new_username, "researcher", "USER_SIGNUP_SUCCESS", f"New user account created: {new_username}")
                                st.session_state.show_signup = False # Switch back to login
                            else:
                                st.error("Failed to create account. Please try again.")
                                log_audit_event(new_username, "researcher", "USER_SIGNUP_FAILURE", f"Failed to save new user account: {new_username} after validation.")
                
                if st.button("Back to Login", key="back_to_login_button"):
                    st.session_state.show_signup = False
                    st.rerun()
            else:
                st.subheader("Login")
                username_input = st.text_input("Username", key="login_username")
                password_input = st.text_input("Password", type="password", key="login_password")
                
                if st.button("Login", key="login_button"):
                    users = load_users()
                    user_data = users.get(username_input, {})
                    if user_data and verify_password(password_input, user_data.get("password", "")):
                        st.session_state.authenticated = True
                        st.session_state.username = username_input
                        # Load user-specific prompt, fallback to role-based, then global default
                        user_specific_prompt = user_data.get("system_prompt")
                        current_role = user_data.get("role", "researcher") # Default role if not specified
                        st.session_state.role = current_role # Store role in session state

                        if not user_specific_prompt:
                            user_specific_prompt = DEFAULT_PROMPTS.get(current_role, SYSTEM_PROMPT)
                        
                        st.session_state.system_prompt = user_specific_prompt
                        st.success("Login successful!")
                        log_audit_event(username_input, current_role, "USER_LOGIN_SUCCESS", f"User {username_input} logged in successfully.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                        log_audit_event(username_input or "UNKNOWN_USER", "N/A", "USER_LOGIN_FAILURE", f"Failed login attempt for username: '{username_input}'.")
                
                if st.button("Create Account", key="show_signup_button"):
                    st.session_state.show_signup = True
                    st.rerun()
        else:
            st.write(f"Logged in as: {st.session_state.username}")
            if st.button("Logout", key="logout_button"):
                logged_out_username = st.session_state.username
                logged_out_role = st.session_state.get("role", "N/A")
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.role = None # Clear role on logout
                st.session_state.system_prompt = SYSTEM_PROMPT # Reset to global default on logout
                st.session_state.show_signup = False # Reset signup view on logout
                log_audit_event(logged_out_username, logged_out_role, "USER_LOGOUT", f"User {logged_out_username} logged out.")
                st.rerun()

            st.markdown("---") # Separator
            st.subheader("Session System Prompt")
            # Allow editing of the session system prompt
            new_prompt = st.text_area(
                "Edit current session's system prompt:", 
                value=st.session_state.system_prompt, 
                height=300,
                key="session_prompt_editor"
            )
            if new_prompt != st.session_state.system_prompt:
                st.session_state.system_prompt = new_prompt
                st.success("Session system prompt updated.")
                log_audit_event(
                    username=st.session_state.username, 
                    role=st.session_state.get("role", "N/A"), 
                    action="SESSION_PROMPT_UPDATED", 
                    details=f"User updated session prompt. New prompt: '{new_prompt}'"
                )

    # Main content area
    if st.session_state.authenticated:
        st.title("AI Research Agent")
        openrouter_client, firecrawl_client = init_clients()

        st.header("Unified Research Interface")

        # --- Add Model Selection Section ---
        st.subheader("Model Selection")
        # Determine the index of the currently configured primary model for default selection
        try:
            default_model_identifier = OPENROUTER_PRIMARY_MODEL # Loaded from config/env
            default_index = MODEL_IDENTIFIERS.index(default_model_identifier)
        except ValueError:
            default_index = 0 # Fallback to the first model if the default isn't in our list
            st.warning(f"Default primary model '{OPENROUTER_PRIMARY_MODEL}' not found in selectable options. Defaulting to first option.")

        selected_model_display_name = st.selectbox(
            "Choose the AI model for report generation:",
            options=MODEL_DISPLAY_NAMES,
            index=default_index,
            key="model_selector",
            help="Select the AI model to use. Defaults are set in config/env."
        )
        # Store the corresponding identifier in session state
        st.session_state.selected_model = MODEL_OPTIONS[selected_model_display_name]
        st.markdown("---") # Separator
        # --- End Model Selection Section ---

        # Research Query Input
        st.subheader("1. Define Your Research Focus")
        research_query = st.text_area(
            "Enter your research query or specific questions:",
            height=100,
            key="research_query_input",
            help="Clearly state what you want the AI to investigate or analyze based on the provided documents and URLs."
        )

        # Document Upload Section
        st.subheader("2. Upload Relevant Documents (Optional)")
        
        uploaded_files_new = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            key="unified_file_uploader", # Using a consistent key might be okay if Streamlit handles updates well
            help="Upload whitepapers, reports, articles, or any other relevant documents."
        )

        # Check if the set of uploaded files has actually changed
        current_file_details = [(f.name, f.size) for f in uploaded_files_new] if uploaded_files_new else []
        files_have_changed = (current_file_details != st.session_state.get("last_uploaded_file_details", []))

        if uploaded_files_new and files_have_changed:
            st.session_state.last_uploaded_file_details = current_file_details
            st.session_state.processed_documents_content = [] # Clear previous batch to process anew
            current_batch_processed_content = []
            
            # This inner check is a bit redundant given the outer `if uploaded_files_new` but safe.
            # if uploaded_files_new: 
            with st.status(f"Processing {len(uploaded_files_new)} uploaded file(s)...", expanded=True) as status_container:
                # progress_bar = st.progress(0) # Using st.status handles ongoing updates better for this pattern
                
                for i, uploaded_file_data in enumerate(uploaded_files_new):
                    st.write(f"Processing: {uploaded_file_data.name} ({i+1}/{len(uploaded_files_new)})")
                    # action_details = f"File upload attempt: {uploaded_file_data.name}" # Audit log moved
                    # log_status = "SUCCESS"

                    file_bytes = uploaded_file_data.getvalue()
                    content = ""
                    # Determine file type for processing (prefer extension for clarity)
                    file_extension = uploaded_file_data.name.split('.')[-1].lower()

                    if file_extension == "pdf":
                        content = extract_text_from_pdf(file_bytes)
                    elif file_extension == "docx":
                        content = extract_text_from_docx(file_bytes)
                    elif file_extension in ["txt", "md"]:
                        content = extract_text_from_txt_md(file_bytes)
                    else:
                        st.warning(f"Unsupported file type skipped (extension not matched): {uploaded_file_data.name}")
                        content = "" 
                        # log_status = "SKIPPED_UNSUPPORTED"

                    if content:
                        current_batch_processed_content.append({"name": uploaded_file_data.name, "text": content})
                        st.success(f"Successfully extracted text from: {uploaded_file_data.name}")
                        action_details = f"Successfully extracted content from uploaded file: {uploaded_file_data.name}"
                        log_audit_event(st.session_state.username, st.session_state.get('role','N/A'), "FILE_PROCESS_SUCCESS", action_details)
                    elif file_extension in ["pdf", "docx", "txt", "md"]:
                        st.error(f"Failed to extract text from: {uploaded_file_data.name} (see specific error above if any).")
                        action_details = f"Failed to extract content from uploaded file: {uploaded_file_data.name}"
                        log_audit_event(st.session_state.username, st.session_state.get('role','N/A'), "FILE_PROCESS_FAILURE", action_details)
                    else: # If skipped due to unsupported type and no content
                        action_details = f"Skipped unsupported file type: {uploaded_file_data.name}"
                        log_audit_event(st.session_state.username, st.session_state.get('role','N/A'), "FILE_PROCESS_SKIPPED", action_details)

                    # progress_bar.progress((i + 1) / len(uploaded_files_new)) # Handled by st.status context
                
                st.session_state.processed_documents_content = current_batch_processed_content
                status_container.update(label=f"{len(current_batch_processed_content)} out of {len(uploaded_files_new)} files processed successfully.", state="complete", expanded=False)
                # Rerun to update displays based on newly processed files in session_state
                # This helps ensure UI consistency immediately after processing
                st.rerun() 
        elif not uploaded_files_new: # If no files are selected (uploader is cleared)
            if st.session_state.last_uploaded_file_details: # If there were files before, clear details
                st.session_state.last_uploaded_file_details = []
                st.session_state.processed_documents_content = []
                st.rerun() # Rerun to reflect that no files are processed

        # Display summary of currently processed documents from session_state
        if st.session_state.get("processed_documents_content"):
            st.markdown("---")
            st.subheader(f"Processed Documents ({len(st.session_state.processed_documents_content)} ready for report):")
            for doc_info in st.session_state.processed_documents_content:
                with st.expander(f"{doc_info['name']} ({len(doc_info['text'])} chars) - Click to preview"):
                    st.caption(doc_info['text'][:250] + "..." if len(doc_info['text']) > 250 else doc_info['text'])
            st.markdown("---")

        # URL Input Section
        st.subheader("3. Provide Web URLs (Optional)")
        st.write("Enter up to 10 URLs for web content scraping:")
        
        # Create a list to hold URL inputs
        url_inputs = [] 
        for i in range(10):
            url = st.text_input(f"URL {i+1}", key=f"url_input_{i+1}", placeholder=f"https://example.com/page{i+1}")
            url_inputs.append(url)
        
        # Collect provided URLs
        submitted_urls = [url for url in url_inputs if url] # Collect non-empty URLs from the list

        # Report Generation
        st.subheader("4. Generate Report")
        if st.button("Generate Unified Report", key="generate_unified_report_button"):
            if not research_query and not uploaded_files_new and not submitted_urls:
                st.warning("Please provide a research query, upload documents, or enter URLs to generate a report.")
            else:
                # Clear previous report content before generating a new one
                st.session_state.unified_report_content = ""
                st.session_state.scraped_web_content = [] # Clear previous scrape results

                # Log the report generation attempt
                processed_doc_names = [doc['name'] for doc in st.session_state.processed_documents_content]
                details_str = f"Research Query: '{research_query}'. Files: {len(processed_doc_names)} ({', '.join(processed_doc_names[:3])}{'...' if len(processed_doc_names) > 3 else ''}). URLs: {len(submitted_urls)}."

                log_audit_event(
                    username=st.session_state.username,
                    role=st.session_state.get('role','N/A'), 
                    action="REPORT_GENERATION_INITIATED",
                    details=details_str,
                    links=submitted_urls if submitted_urls else None
                )

                with st.spinner("Processing inputs and generating report..."):
                    # --- Stage 1: Scrape Web Content (Task 3) ---
                    if submitted_urls:
                        st.info(f"Starting web scraping for {len(submitted_urls)} URL(s)...")
                        scraped_data = await process_urls(submitted_urls, firecrawl_client)
                        st.session_state.scraped_web_content = scraped_data
                        
                        # Display scraping feedback
                        successful_scrapes = 0
                        for item in scraped_data:
                            if item["status"] == "success":
                                st.success(f"Successfully scraped: {item['url']}")
                                # Audit log for successful scrape included in REPORT_GENERATION_INITIATED with all links
                                # log_audit_event(st.session_state.username, st.session_state.get('role','N/A'), "URL_SCRAPE_SUCCESS", f"Scraped: {item['url']}", links=[item['url']])
                                successful_scrapes += 1
                            else:
                                st.warning(f"Failed to scrape: {item['url']} - Error: {item['error']}")
                                log_audit_event(st.session_state.username, st.session_state.get('role','N/A'), "URL_SCRAPE_FAILURE", f"Failed to scrape: {item['url']}. Error: {item['error']}", links=[item['url']])
                        if successful_scrapes > 0:
                            st.info(f"Finished scraping: {successful_scrapes}/{len(submitted_urls)} URLs scraped successfully.")
                        else:
                            st.warning("Web scraping completed, but no content was successfully retrieved from the provided URLs.")
                    else:
                        st.info("No URLs provided for web scraping.")
                    
                    # --- Task 4: Combined Analysis & Report Generation --- 
                    st.info("Combining processed content and generating AI report...")
                    
                    # 1. Retrieve research query
                    # research_query variable is already available here

                    # 2. Retrieve text from processed documents
                    document_texts = []
                    if st.session_state.processed_documents_content:
                        for doc in st.session_state.processed_documents_content:
                            document_texts.append(f"--- Document: {doc['name']} ---\n{doc['text']}\n---")
                    combined_document_text = "\n".join(document_texts)

                    # 3. Retrieve text from scraped URLs
                    scraped_content_texts = []
                    if st.session_state.scraped_web_content:
                        for item in st.session_state.scraped_web_content:
                            if item["status"] == "success" and item.get("content"): # Ensure content exists
                                scraped_content_texts.append(f"--- Web Content from: {item['url']} ---\n{item['content']}\n---")
                    combined_scraped_text = "\n".join(scraped_content_texts)

                    # 4. Combine all text sources with the research query
                    # The research_query acts as the primary instruction/question.
                    # The system_prompt (in st.session_state.system_prompt) guides the AI's persona and output format.
                    
                    # Construct the main instruction based on whether a user query was provided
                    if research_query:
                        full_prompt_for_ai = f"Research Query: {research_query}\\n\\n"
                    else:
                        # Default instruction if query is empty - rely on system prompt and content
                        full_prompt_for_ai = "Research Goal: Please generate a comprehensive report based on the provided content (if any) and the overall objectives defined in the system prompt.\\n\\n"

                    # Append document content if available
                    if combined_document_text:
                        full_prompt_for_ai += f"Provided Document(s) Content:\\n{combined_document_text}\\n\\n"
                    else:
                        full_prompt_for_ai += "No documents were provided or processed.\\n\\n"
                        
                    # Append scraped web content if available
                    if combined_scraped_text:
                        full_prompt_for_ai += f"Provided Web Content:\\n{combined_scraped_text}\\n\\n"
                    else:
                        full_prompt_for_ai += "No web content was provided or successfully scraped.\\n\\n"
                    
                    # Final instruction part
                    full_prompt_for_ai += "Based on the research goal/query and all the provided content above, please generate a comprehensive report."

                    # 5. Call OpenRouterClient
                    # Ensure openrouter_client is initialized (it is, outside this button block)
                    
                    # --- DEBUGGING --- 
                    st.write("--- DEBUG INFO BEFORE AI CALL ---")
                    # Prepare query snippet for debugging, handling potential quotes
                    debug_query_snippet = research_query[:50]
                    if len(research_query) > 50:
                        debug_query_snippet += "..."
                    st.write(f"Research Query Provided: {'Yes' if research_query else 'No'} (Content: '{debug_query_snippet}')") # Use prepared snippet
                    st.write(f"Combined Document Text Provided: {'Yes' if combined_document_text else 'No'} (Length: {len(combined_document_text)})")
                    st.write(f"Combined Scraped Text Provided: {'Yes' if combined_scraped_text else 'No'} (Length: {len(combined_scraped_text)})")
                    st.write("--- END DEBUG INFO ---")
                    # --- END DEBUGGING ---

                    try:
                        # --- Get the selected model ---
                        model_to_use = st.session_state.get("selected_model", OPENROUTER_PRIMARY_MODEL) # Fallback just in case
                        log_audit_event(st.session_state.username, st.session_state.get('role','N/A'), "MODEL_SELECTION_USED", f"Using model: {model_to_use} for report generation.")
                        # --- End Get model ---

                        ai_generated_report = await openrouter_client.generate_response(
                            prompt=full_prompt_for_ai, 
                            system_prompt=st.session_state.system_prompt,
                            model_override=model_to_use # Pass selected model
                        )
                        if ai_generated_report:
                            st.session_state.unified_report_content = ai_generated_report
                            st.success("AI Report generated successfully!")
                            
                            successfully_scraped_urls = [item['url'] for item in st.session_state.scraped_web_content if item["status"] == "success"]
                            query_for_log = research_query if research_query else '[SYSTEM PROMPT]'
                            success_details = f"AI report generated for query: '{query_for_log}'. Docs: {len(processed_doc_names)}. Scraped URLs: {len(successfully_scraped_urls)}."
                            
                            log_audit_event(
                                username=st.session_state.username, 
                                role=st.session_state.get('role','N/A'), 
                                action="REPORT_GENERATION_SUCCESS", 
                                details=success_details,
                                links=successfully_scraped_urls if successfully_scraped_urls else None
                            )
                        else:
                            st.session_state.unified_report_content = "Failed to generate AI report. The AI returned an empty response."
                            st.error("AI report generation failed or returned empty.")
                            query_for_log = research_query if research_query else '[SYSTEM PROMPT]'
                            log_audit_event(username=st.session_state.username, role=st.session_state.get('role','N/A'), action="REPORT_GENERATION_FAILURE", details=f"AI returned empty report for query: '{query_for_log}'")
                    except Exception as e:
                        st.session_state.unified_report_content = f"An error occurred during AI report generation: {str(e)}"
                        st.error(f"Error calling AI: {e}")
                        query_for_log = research_query if research_query else '[SYSTEM PROMPT]'
                        log_audit_event(username=st.session_state.username, role=st.session_state.get('role','N/A'), action="REPORT_GENERATION_ERROR", details=f"Error during AI call for query '{query_for_log}': {str(e)}")
                    
                    # Remove the debug JSON output now that real processing is in place
                    # st.markdown("--- DEBUG: Information Collected ---\")
                    # st.json({
                    #     "research_query": research_query,
                    #     "processed_documents": st.session_state.processed_documents_content,
                    #     "scraped_urls_content": st.session_state.scraped_web_content
                    # })
                    # st.info("Report generation logic (Task 4) is pending. Displaying collected data for now.")
                    # --- End of Task 4 ---
                
                # Rerun to display the generated report or messages
                st.rerun()
        
        # Area for actual report display (will be populated by backend logic)
        if "unified_report_content" in st.session_state and st.session_state.unified_report_content:
            st.markdown("---")
            st.subheader("Generated Report")
            st.markdown(st.session_state.unified_report_content)
            st.download_button(
                label="Download Unified Report",
                data=st.session_state.unified_report_content,
                file_name="unified_research_report.md",
                mime="text/markdown",
                key="download_actual_unified_report",
                on_click=lambda: log_audit_event(
                    username=st.session_state.username, 
                    role=st.session_state.get('role', 'N/A'), 
                    action="REPORT_DOWNLOADED", 
                    details=f"User downloaded report: unified_research_report.md for query '{(st.session_state.get('research_query_input', 'QUERY_NOT_IN_SESSION_FOR_DOWNLOAD_LOG'))}'"
                )
            )
        # elif "generate_unified_report_button_clicked" in st.session_state and \
        #      st.session_state.generate_unified_report_button_clicked and \
        #      not st.session_state.unified_report_content:
        #      # This logic might be implicitly handled by the spinner and rerun, 
        #      # or if a specific message for "no report generated" is desired, it can be placed here.
        #      # For now, if content is empty, nothing is shown, which is acceptable.
        #      pass 

    else: # Not authenticated
        st.info("Please log in or create an account to access the research pipeline.")
        # The login/signup logic from the sidebar already handles this visibility.
        # This message is a fallback or main page content for unauthenticated users.

async def run_main():
    await main()

if __name__ == "__main__":
    asyncio.run(run_main()) 