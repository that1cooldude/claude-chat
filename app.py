import streamlit as st 
import boto3
import json
from datetime import datetime
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Page config
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional CSS
st.markdown("""
<style>
    .message-container {
        margin: 15px 0;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .user-message { 
        background-color: #2e3136; 
        margin-left: 20px;
        border-left: 3px solid #5865F2;
    }
    .assistant-message { 
        background-color: #36393f; 
        margin-right: 20px;
        border-left: 3px solid #43B581;
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
    .control-button {
        background-color: transparent;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .control-button:hover {
        background-color: rgba(255,255,255,0.1);
    }
    @media (max-width: 768px) {
        .message-container { 
            margin: 10px 5px; 
            padding: 12px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    """Get response from Claude with thinking process"""
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

@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock client"""
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

def process_message(message: str, role: str, thinking: str = None) -> dict:
    """Format message with metadata"""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "thinking": thinking
    }

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
    
    # Chat Management
    st.subheader("Conversations")
    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create Chat") and new_chat_name:
        if new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "settings": {"temperature": 0.7, "max_tokens": 1000},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.current_chat = new_chat_name
            st.rerun()
    
    st.session_state.current_chat = st.selectbox(
        "Select Chat", 
        options=list(st.session_state.chats.keys())
    )
    
    # Model Settings
    st.subheader("Model Settings")
    current_chat = st.session_state.chats[st.session_state.current_chat]
    current_chat["settings"]["temperature"] = st.slider(
        "Temperature", 
        min_value=0.0,
        max_value=1.0,
        value=current_chat["settings"]["temperature"],
        help="Higher values make responses more creative, lower values make them more focused"
    )
    current_chat["settings"]["max_tokens"] = st.slider(
        "Max Tokens", 
        min_value=100,
        max_value=4096,
        value=current_chat["settings"]["max_tokens"],
        help="Maximum length of the response"
    )
    
    # Display Settings
    st.subheader("Display Settings")
    st.session_state.show_thinking = st.toggle(
        "Show Thinking Process",
        value=st.session_state.show_thinking,
        help="Show or hide Claude's reasoning process"
    )
    
    # Chat Management
    st.subheader("Chat Management")
    if st.button("Clear Current Chat", help="Delete all messages in current chat"):
        if st.session_state.current_chat in st.session_state.chats:
            st.session_state.chats[st.session_state.current_chat]["messages"] = []
            st.rerun()

# Main chat interface
st.title(f"Claude Chat - {st.session_state.current_chat}")

# Message display
for idx, message in enumerate(current_chat["messages"]):
    with st.chat_message(message["role"]):
        if st.session_state.editing_message == idx and message["role"] == "user":
            # Edit mode
            edited_message = st.text_area(
                "Edit message",
                message["content"],
                key=f"edit_{idx}"
            )
            col1, col2 = st.columns([1,4])
            with col1:
                if st.button("Save", key=f"save_{idx}"):
                    current_chat["messages"][idx]["content"] = edited_message
                    st.session_state.editing_message = None
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_{idx}"):
                    st.session_state.editing_message = None
                    st.rerun()
        else:
            # Normal message display
            st.markdown(message["content"])
            st.caption(f"Time: {message['timestamp']}")
            
            # Message controls for user messages
            if message["role"] == "user":
                col1, col2 = st.columns([1,20])
                with col1:
                    if st.button("✏️", key=f"edit_{idx}", help="Edit message"):
                        st.session_state.editing_message = idx
                        st.rerun()
                with col1:
                    if st.button("🗑️", key=f"delete_{idx}", help="Delete message"):
                        current_chat["messages"].pop(idx)
                        st.rerun()
            
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
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    current_chat["messages"].append(process_message(prompt, "user"))
    
    # Get Claude response
    client = get_bedrock_client()
    if client:
        response = get_chat_response(
            prompt,
            current_chat["messages"][-5:],
            client,
            current_chat["settings"]
        )
        
        if response:
            thinking, main_response = response
            current_chat["messages"].append(
                process_message(main_response, "assistant", thinking=thinking)
            )
            st.rerun()
