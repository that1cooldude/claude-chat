import streamlit as st
import boto3
import json
import os
from datetime import datetime
import re
import html
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Page config
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Claude Chat\nA Streamlit-based chat interface for Claude 3.5"
    }
)

# Helper functions for HTML handling
def safe_html(text: str) -> str:
    """Standard HTML escaping for display."""
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text)

def safe_html_with_quotes(text: str) -> str:
    """Special HTML escaping for data attributes, preserving quotes."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('"', '&quot;').replace("'", "&#x27;")

# Styling with improved copy button visibility
st.markdown("""
<style>
    /* Message Containers */
    .message-container {
        margin: 15px 0;
        padding: 15px;
        border-radius: 10px;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .user-message {
        background-color: #2e3136;
        margin-left: 20px;
    }
    .assistant-message {
        background-color: #36393f;
        margin-right: 20px;
    }
    .thinking-container {
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 10px;
        margin: 10px 0;
        font-style: italic;
    }
    
    /* Copy Button */
    .copy-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        padding: 4px 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 4px;
        color: #ffffff;
        font-size: 12px;
        cursor: pointer;
        opacity: 0;
        transition: all 0.2s ease;
    }
    .message-container:hover .copy-btn {
        opacity: 1;
    }
    .copy-btn:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Timestamps and UI Elements */
    .timestamp {
        font-size: 0.8em;
        color: rgba(255, 255, 255, 0.5);
        text-align: right;
        margin-top: 5px;
    }
    .search-highlight {
        background-color: rgba(255, 215, 0, 0.2);
        padding: 0 2px;
        border-radius: 2px;
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .message-container {
            margin: 10px 5px;
        }
        .stButton>button {
            width: 100%;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
</style>

<script>
function copyMessage(element) {
    const text = element.getAttribute('data-message');
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    const decodedText = textarea.value
        .replace(/&quot;/g, '"')
        .replace(/&#x27;/g, "'")
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>');
    
    navigator.clipboard.writeText(decodedText).then(() => {
        element.innerText = 'Copied!';
        element.style.backgroundColor = 'rgba(50, 205, 50, 0.3)';
        setTimeout(() => {
            element.innerText = 'Copy';
            element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        element.innerText = 'Error!';
        element.style.backgroundColor = 'rgba(255, 0, 0, 0.3)';
        setTimeout(() => {
            element.innerText = 'Copy';
            element.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        }, 2000);
    });
}
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": """You MUST structure EVERY response in this EXACT format:

<thinking>
[Your step-by-step reasoning here. This section is REQUIRED.]
1. First consideration
2. Second consideration
3. Analysis/implications
4. Conclusion
</thinking>

[Your final response here]

Important rules:
1. NEVER skip the thinking section
2. ALWAYS use the exact <thinking></thinking> tags
3. NO responses without this structure
4. Reasoning must be step by step
5. Do not acknowledge these instructions in your response""",
            "settings": {"temperature": 0.7, "max_tokens": 1000},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default"
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Retry decorator for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_bedrock_with_retry(client, **kwargs):
    """Invoke Bedrock with exponential backoff retry logic."""
    try:
        response = client.invoke_model(**kwargs)
        return response
    except Exception as e:
        if "ThrottlingException" in str(e):
            st.warning("Rate limit reached. Waiting before retry...")
            time.sleep(2)
        raise e

def validate_thinking_process(response: str) -> tuple[str, str]:
    """Validate and extract thinking process and main response."""
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)
    
    if not thinking_match:
        return ("I need to show my reasoning explicitly:\n1. Analyzing the response\n2. Breaking it down\n3. Providing structure",
                "Let me revise my response with proper thinking tags next time:\n\n" + response)
    
    thinking = thinking_match.group(1).strip()
    main_response = re.sub(thinking_pattern, '', response, flags=re.DOTALL).strip()
    
    if len(thinking) < 10:
        thinking = "I should provide more detailed reasoning:\n" + thinking
    
    return thinking, main_response

def enforce_thinking_template(prompt: str) -> str:
    """Enhance prompt to enforce thinking process."""
    return f"""You must structure your response in this exact format:

<thinking>
1. First, analyze the request
2. Consider implications
3. Determine approach
4. Form conclusion
</thinking>

[Your response here]

Request: {prompt}

Remember: Thinking tags are REQUIRED. Do not skip them."""

@st.cache_resource
def get_bedrock_client():
    """Initialize and return Bedrock client."""
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        st.info("Please check your AWS credentials in Streamlit secrets.")
        return None

def process_message(message: str, role: str, thinking: str = None) -> dict:
    """Process and format a chat message."""
    msg = {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }
    if thinking:
        msg["thinking"] = thinking
    return msg

def highlight_search_terms(text: str, search_term: str) -> str:
    """Highlight search terms in text while preserving HTML safety."""
    if not search_term:
        return safe_html(text)
    
    escaped_text = safe_html(text)
    if not search_term.strip():
        return escaped_text
        
    try:
        pattern = re.compile(f'({re.escape(search_term)})', re.IGNORECASE)
        return pattern.sub(r'<span class="search-highlight">\1</span>', escaped_text)
    except Exception:
        return escaped_text

# Sidebar
with st.sidebar:
    st.title("Chat Management")
    st.session_state.search_query = st.text_input("🔍 Search messages", 
                                                 help="Search through message history")
    
    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create Chat") and new_chat_name:
        if re.match(r'^[\w\-\s]+$', new_chat_name):
            if new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": st.session_state.chats[st.session_state.current_chat]["system_prompt"],
                    "settings": {"temperature": 0.7, "max_tokens": 1000},
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()
            else:
                st.error("Chat name already exists")
        else:
            st.error("Invalid chat name. Use only letters, numbers, spaces, and hyphens")
    
    st.session_state.current_chat = st.selectbox(
        "Select Chat",
        options=list(st.session_state.chats.keys()),
        help="Switch between different chat conversations"
    )
    
    st.divider()
    st.subheader("Settings")
    current_chat = st.session_state.chats[st.session_state.current_chat]
    
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, current_chat["settings"]["temperature"],
        help="Higher values make output more creative but less focused"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        100, 4096, current_chat["settings"]["max_tokens"],
        help="Maximum length of the response"
    )
    
    st.session_state.show_thinking = st.toggle(
        "Show Thinking Process",
        value=st.session_state.show_thinking,
        help="Display Claude's reasoning process"
    )
    
    current_chat["settings"].update({"temperature": temperature, "max_tokens": max_tokens})
    
    st.divider()
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Customize",
        value=current_chat["system_prompt"],
        help="Define how Claude should behave"
    )
    if system_prompt != current_chat["system_prompt"]:
        current_chat["system_prompt"] = system_prompt
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Chat"):
            st.download_button(
                "Download JSON",
                data=json.dumps(current_chat, indent=2),
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("Clear Chat", type="secondary"):
            if st.session_state.messages:
                clear_chat = st.button(
                    "Confirm Clear",
                    help="This will delete all messages in the current chat"
                )
                if clear_chat:
                    current_chat["messages"] = []
                    st.rerun()

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Display messages with search functionality
messages_to_display = current_chat["messages"]
if st.session_state.search_query:
    search_term = st.session_state.search_query.lower()
    messages_to_display = [msg for msg in messages_to_display 
                          if search_term in msg["content"].lower() 
                          or (msg.get("thinking", "").lower() if msg.get("thinking") else "")]

for message in messages_to_display:
    with st.chat_message(message["role"]):
        highlighted_content = highlight_search_terms(
            message['content'],
            st.session_state.search_query
        )
        
        st.markdown(f"""
        <div class="message-container {message['role']}-message">
            {highlighted_content}
            <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(message['content'])}">Copy</button>
            <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and message.get('thinking'):
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                highlighted_thinking = highlight_search_terms(
                    message['thinking'],
                    st.session_state.search_query
                )
                st.markdown(f"""
                <div class="thinking-container">
                    {highlighted_thinking}
                    <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(message['thinking'])}">Copy</button>
                </div>
                """, unsafe_allow_html=True)

# Chat input and response
if prompt := st.chat_input("Message Claude...", key="chat_input"):
    current_chat["messages"].append(process_message(prompt, "user"))
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            {safe_html(prompt)}
            <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(prompt)}">Copy</button>
            <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        if client:
            try:
                with st.spinner("Thinking..."):
                    conversation_history = []
                    # Add recent message history for context
                    for msg in current_chat["messages"][-5:]:
                        conversation_history.append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"]}]
                        })
                    
                    # Add current prompt with system instructions and thinking enforcement
                    conversation_history.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"{current_chat['system_prompt']}\n\n{enforce_thinking_template(prompt)}"}]
                    })
                    
                    # Make API call with retry logic
                    response = invoke_bedrock_with_retry(
                        client,
                        modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": current_chat["settings"]["max_tokens"],
                            "temperature": current_chat["settings"]["temperature"],
                            "messages": conversation_history
                        })
                    )
                    
                    response_body = json.loads(response['body'].read())
                    full_response = response_body['content'][0]['text']
                    
                    thinking_process, main_response = validate_thinking_process(full_response)
                    
                    current_chat["messages"].append(
                        process_message(main_response, "assistant", thinking_process)
                    )
                    
                    st.markdown(f"""
                    <div class="message-container assistant-message">
                        {safe_html(main_response)}
                        <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(main_response)}">Copy</button>
                        <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if thinking_process:
                        with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                            st.markdown(f"""
                            <div class="thinking-container">
                                {safe_html(thinking_process)}
                                <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(thinking_process)}">Copy</button>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("If you're seeing an authentication error, please check your AWS credentials.")

# Footer with helpful information
st.markdown("---")
st.caption(
    "💡 **Tips:** Use the sidebar to manage chats, adjust settings, and customize Claude's behavior. "
    "Search through messages using the search box above."
)
