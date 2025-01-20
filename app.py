import streamlit as st 
import boto3
import json
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Page config
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
<style>
    .stChat message {
        background-color: #2e3136;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .thinking-container {
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 15px;
        margin: 10px 0;
        font-style: italic;
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
def get_chat_response(message: str, client, settings: dict):
    """Get response from Claude using correct message format"""
    try:
        with st.spinner("Thinking..."):
            response = client.invoke_model(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "top_k": 250,
                    "top_p": 0.999,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": message
                                }
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body.get('content', '')
            
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None

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

def process_message(message: str, role: str) -> dict:
    """Format message with metadata"""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p')
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
        0.0, 1.0,
        current_chat["settings"]["temperature"]
    )
    current_chat["settings"]["max_tokens"] = st.slider(
        "Max Tokens",
        100, 4096,
        current_chat["settings"]["max_tokens"]
    )
    
    # Clear Chat
    if st.button("Clear Current Chat"):
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
            cols = st.columns([1,4])
            with cols[0]:
                if st.button("Save", key=f"save_{idx}"):
                    current_chat["messages"][idx]["content"] = edited_message
                    st.session_state.editing_message = None
                    st.rerun()
            with cols[1]:
                if st.button("Cancel", key=f"cancel_{idx}"):
                    st.session_state.editing_message = None
                    st.rerun()
        else:
            # Normal message display
            st.markdown(message["content"])
            st.caption(f"Time: {message['timestamp']}")
            
            # Message controls for user messages
            if message["role"] == "user":
                cols = st.columns([1,1,10])
                with cols[0]:
                    if st.button("✏️", key=f"edit_{idx}"):
                        st.session_state.editing_message = idx
                        st.rerun()
                with cols[1]:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        current_chat["messages"].pop(idx)
                        st.rerun()

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    current_chat["messages"].append(process_message(prompt, "user"))
    
    # Get Claude response
    client = get_bedrock_client()
    if client:
        response = get_chat_response(
            prompt,
            client,
            current_chat["settings"]
        )
        
        if response:
            current_chat["messages"].append(
                process_message(response, "assistant")
            )
            st.rerun()
