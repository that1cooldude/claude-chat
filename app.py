import streamlit as st
import boto3
import json
import os
from datetime import datetime
import time
import re

# Page configuration
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling with copy functionality
st.markdown("""
<style>
    /* Base Styles */
    .message-container {
        margin: 15px 0;
        padding: 10px;
        border-radius: 15px;
        position: relative;
    }
    .user-message {
        background-color: #2e3136;
        margin-left: 20px;
        border: 1px solid #404040;
    }
    .assistant-message {
        background-color: #36393f;
        margin-right: 20px;
        border: 1px solid #404040;
    }
    .thinking-container {
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 10px;
        margin: 10px 0;
        font-style: italic;
        color: #b8b8b8;
    }
    .timestamp {
        font-size: 0.8em;
        color: #666;
        text-align: right;
    }
    /* Copy Button */
    .copy-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        padding: 2px 8px;
        background: #404040;
        border: none;
        border-radius: 3px;
        color: white;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .message-container:hover .copy-btn {
        opacity: 1;
    }
    /* Search Highlight */
    .search-highlight {
        background-color: #ffd70066;
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
</style>

<script>
function copyMessage(element) {
    const text = element.getAttribute('data-message');
    navigator.clipboard.writeText(text);
    element.innerText = 'Copied!';
    setTimeout(() => {
        element.innerText = 'Copy';
    }, 2000);
}
</script>
""", unsafe_allow_html=True)

# Initialize session state with documentation
def init_session_state():
    """Initialize or update session state variables."""
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Default": {
                "messages": [],
                "system_prompt": "You are a helpful AI assistant. Always show your reasoning using <thinking></thinking> tags before providing your final response.",
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

init_session_state()

# Helper functions with documentation
def validate_chat_name(name: str) -> tuple[bool, str]:
    """
    Validate chat name against allowed patterns.
    
    Args:
        name (str): The chat name to validate
        
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not name:
        return False, "Chat name cannot be empty"
    if not re.match(r'^[\w\-\s]+$', name):
        return False, "Chat name can only contain letters, numbers, spaces, and hyphens"
    return True, ""

def process_message(message: str, role: str) -> dict:
    """
    Process and format a chat message.
    
    Args:
        message (str): The message content
        role (str): The role ('user' or 'assistant')
        
    Returns:
        dict: Formatted message object
    """
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }

def highlight_search_text(text: str, search_query: str) -> str:
    """
    Highlight search query matches in text.
    
    Args:
        text (str): The text to search in
        search_query (str): The search query
        
    Returns:
        str: Text with search highlights
    """
    if not search_query:
        return text
    pattern = re.compile(f'({re.escape(search_query)})', re.IGNORECASE)
    return pattern.sub(r'<span class="search-highlight">\1</span>', text)

# Bedrock client initialization - preserved working setup
@st.cache_resource
def get_bedrock_client():
    """
    Initialize and return a cached Bedrock client instance.
    
    Returns:
        boto3.client: Configured Bedrock client
    """
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=st.secrets["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

# Sidebar functionality
with st.sidebar:
    st.title("Chat Management")
    
    # Search functionality
    st.session_state.search_query = st.text_input("🔍 Search messages", value=st.session_state.search_query)
    
    # Chat creation with validation
    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create Chat") and new_chat_name:
        is_valid, error_msg = validate_chat_name(new_chat_name)
        if is_valid:
            if new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": st.session_state.chats[st.session_state.current_chat]["system_prompt"],
                    "settings": {
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.success(f"Created chat: {new_chat_name}")
                st.rerun()
            else:
                st.error("Chat name already exists")
        else:
            st.error(error_msg)
    
    # Chat selection
    st.session_state.current_chat = st.selectbox(
        "Select Chat",
        options=list(st.session_state.chats.keys())
    )
    
    # System prompt
    st.divider()
    system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.chats[st.session_state.current_chat]["system_prompt"]
    )
    if system_prompt != st.session_state.chats[st.session_state.current_chat]["system_prompt"]:
        st.session_state.chats[st.session_state.current_chat]["system_prompt"] = system_prompt
    
    # Settings
    st.divider()
    temperature = st.slider("Temperature", 0.0, 1.0, 
                          st.session_state.chats[st.session_state.current_chat]["settings"]["temperature"])
    max_tokens = st.slider("Max Tokens", 100, 4096,
                          st.session_state.chats[st.session_state.current_chat]["settings"]["max_tokens"])
    show_thinking = st.toggle("Show Thinking Process", value=st.session_state.show_thinking)
    
    # Update settings
    current_chat = st.session_state.chats[st.session_state.current_chat]
    current_chat["settings"].update({
        "temperature": temperature,
        "max_tokens": max_tokens
    })
    st.session_state.show_thinking = show_thinking
    
    # Chat management buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Chat"):
            chat_data = st.session_state.chats[st.session_state.current_chat]
            st.download_button(
                "Download JSON",
                data=json.dumps(chat_data, indent=2),
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chats[st.session_state.current_chat]["messages"] = []
            st.rerun()

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Display messages with search highlighting and copy functionality
current_messages = st.session_state.chats[st.session_state.current_chat]["messages"]
if st.session_state.search_query:
    current_messages = [msg for msg in current_messages 
                       if st.session_state.search_query.lower() in msg["content"].lower()]

for message in current_messages:
    with st.chat_message(message["role"]):
        # Prepare message content with search highlighting
        content = highlight_search_text(message["content"], st.session_state.search_query)
        
        # Message container with copy button
        st.markdown(f"""
        <div class="message-container {message['role']}-message">
            {content}
            <button class="copy-btn" onclick="copyMessage(this)" data-message="{message['content']}">Copy</button>
            <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display thinking process for assistant messages
        if message['role'] == 'assistant' and 'thinking' in message:
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                thinking_content = highlight_search_text(message['thinking'], st.session_state.search_query)
                st.markdown(f"""
                <div class="thinking-container">
                    {thinking_content}
                    <button class="copy-btn" onclick="copyMessage(this)" data-message="{message['thinking']}">Copy</button>
                </div>
                """, unsafe_allow_html=True)

# Chat input and response handling
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    user_message = process_message(prompt, "user")
    st.session_state.chats[st.session_state.current_chat]["messages"].append(user_message)
    
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            {prompt}
            <div class="timestamp">{user_message['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Claude's response
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        try:
            with st.spinner("Thinking..."):
                # Prepare conversation history
                conversation_history = []
                for msg in st.session_state.chats[st.session_state.current_chat]["messages"]:
                    conversation_history.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                
                # Add current prompt
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                })
                
                # Make API call with conversation history
                response = client.invoke_model(
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
                
                # Extract thinking process
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
                thinking_process = thinking_match.group(1) if thinking_match else ""
                main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
                
                # Store assistant response
                assistant_message = process_message(main_response, "assistant")
                assistant_message["thinking"] = thinking_process
                st.session_state.chats[st.session_state.current_chat]["messages"].append(assistant_message)
                
                # Display response
                st.markdown(f"""
                <div class="message-container assistant-message">
                    {main_response}
                    <button class="copy-btn" onclick="copyMessage(this)" data-message="{main_response}">Copy</button>
                    <div class="timestamp">{assistant_message['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if thinking_process and st.session_state.show_thinking:
                    with st.expander("Thinking Process", expanded=True):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {thinking_process}
                            <button class="copy-btn" onclick="copyMessage(this)" data-message="{thinking_process}">Copy</button>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("If you're seeing an authentication error, please check your AWS credentials.")
