import streamlit as st 
import boto3
import json
from datetime import datetime
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration and page setup
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - Streamlit best practices for styling
st.markdown("""
<style>
    .stChat message {
        background-color: #2e3136 !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 5px 0 !important;
    }
    .thinking-container {
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 15px;
        margin: 10px 0;
        font-style: italic;
        border-radius: 5px;
    }
    .timestamp {
        font-size: 0.8em;
        color: rgba(255,255,255,0.5);
        text-align: right;
        margin-top: 5px;
    }
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .stButton button {
            width: 100%;
            margin: 2px 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state - Streamlit recommended pattern
def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Default": {
                "messages": [],
                "settings": {"temperature": 0.7, "max_tokens": 1000},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True

init_session_state()

# Cached resource for AWS client - Streamlit recommended pattern
@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock client with proper error handling"""
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

# API call with retry logic
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
def get_claude_response(prompt: str, client, settings: dict):
    """Get response from Claude with proper error handling"""
    try:
        with st.spinner("Thinking..."):
            response = client.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=json.dumps({
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "anthropic_version": "bedrock-2023-05-31"
                })
            )
            
            response_body = json.loads(response.get('body').read())
            full_response = response_body.get('completion', '')
            
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
            else:
                thinking = "Reasoning process not explicitly provided"
                main_response = full_response
                
            return thinking, main_response
            
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None, None

def process_message(content: str, role: str, thinking: str = None) -> dict:
    """Process and format a message with metadata"""
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "thinking": thinking
    }

def display_message_controls(idx: int, message: dict, current_chat: dict):
    """Display message control buttons using Streamlit components"""
    if message["role"] == "user":
        cols = st.columns([1, 1, 10])
        with cols[0]:
            if st.button("✏️", key=f"edit_btn_{idx}", help="Edit message"):
                st.session_state.editing_message = idx
                st.rerun()
        with cols[1]:
            if st.button("🗑️", key=f"delete_btn_{idx}", help="Delete message"):
                current_chat["messages"].pop(idx)
                st.rerun()

# Sidebar components
with st.sidebar:
    st.title("Chat Settings")
    
    # Chat Management
    st.subheader("Conversations")
    new_chat_name = st.text_input("New Chat Name", key="new_chat_input")
    if st.button("Create Chat", key="create_chat_btn") and new_chat_name:
        if new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "settings": {"temperature": 0.7, "max_tokens": 1000},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.current_chat = new_chat_name
            st.rerun()
    
    # Chat Selection
    st.session_state.current_chat = st.selectbox(
        "Select Chat",
        options=list(st.session_state.chats.keys()),
        key="chat_selector"
    )
    
    # Model Settings
    current_chat = st.session_state.chats[st.session_state.current_chat]
    st.subheader("Model Settings")
    
    current_chat["settings"]["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=current_chat["settings"]["temperature"],
        help="Higher values make responses more creative",
        key="temperature_slider"
    )
    
    current_chat["settings"]["max_tokens"] = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4096,
        value=current_chat["settings"]["max_tokens"],
        help="Maximum length of the response",
        key="max_tokens_slider"
    )
    
    # Display Settings
    st.subheader("Display Settings")
    st.session_state.show_thinking = st.toggle(
        "Show Thinking Process",
        value=st.session_state.show_thinking,
        help="Show or hide Claude's reasoning process",
        key="thinking_toggle"
    )
    
    # Clear Chat Button
    if st.button("Clear Current Chat", key="clear_chat_btn"):
        if st.session_state.current_chat in st.session_state.chats:
            st.session_state.chats[st.session_state.current_chat]["messages"] = []
            st.rerun()

# Main chat interface
st.title(f"Claude Chat - {st.session_state.current_chat}")

# Message display
for idx, message in enumerate(current_chat["messages"]):
    with st.chat_message(message["role"]):
        # Handle message editing
        if st.session_state.editing_message == idx and message["role"] == "user":
            edited_message = st.text_area(
                "Edit message",
                message["content"],
                key=f"edit_area_{idx}"
            )
            cols = st.columns([1, 1])
            with cols[0]:
                if st.button("Save", key=f"save_btn_{idx}"):
                    current_chat["messages"][idx]["content"] = edited_message
                    st.session_state.editing_message = None
                    st.rerun()
            with cols[1]:
                if st.button("Cancel", key=f"cancel_btn_{idx}"):
                    st.session_state.editing_message = None
                    st.rerun()
        else:
            # Display message content
            st.markdown(message["content"])
            st.caption(f"Time: {message['timestamp']}")
            
            # Display message controls
            display_message_controls(idx, message, current_chat)
            
            # Display thinking process for assistant messages
            if message["role"] == "assistant" and message.get("thinking"):
                if st.session_state.show_thinking:
                    with st.expander("Thinking Process", expanded=True):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {message['thinking']}
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Message Claude...", key="chat_input"):
    # Add user message
    current_chat["messages"].append(process_message(prompt, "user"))
    
    # Get Claude response
    client = get_bedrock_client()
    if client:
        response = get_claude_response(prompt, client, current_chat["settings"])
        
        if response:
            thinking, main_response = response
            current_chat["messages"].append(
                process_message(main_response, "assistant", thinking=thinking)
            )
            st.rerun()
