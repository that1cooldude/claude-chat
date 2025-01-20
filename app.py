import streamlit as st 
import boto3
import json
import os
from datetime import datetime
import re
import html
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO
import csv

# Page config
st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": """You MUST structure EVERY response with thinking and final answer sections.""",
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
if "editing_message" not in st.session_state:
    st.session_state.editing_message = None
if "message_action" not in st.session_state:
    st.session_state.message_action = None
if "favorite_prompts" not in st.session_state:
    st.session_state.favorite_prompts = []
if "reactions" not in st.session_state:
    st.session_state.reactions = {}

# Custom CSS
st.markdown("""
<style>
    .message-container { 
        margin: 15px 0; 
        padding: 15px; 
        border-radius: 10px; 
        position: relative; 
        border: 1px solid rgba(255,255,255,0.1);
        overflow-wrap: break-word;
    }
    .message-content {
        margin-bottom: 10px;
        white-space: pre-wrap;
        max-width: 100%;
    }
    .user-message { background-color: #2e3136; margin-left: 20px; }
    .assistant-message { background-color: #36393f; margin-right: 20px; }
    .thinking-container { 
        background-color: #1e1e2e; 
        border-left: 3px solid #ffd700; 
        padding: 10px; 
        margin: 10px 0; 
        font-style: italic; 
    }
    .timestamp { 
        font-size: 0.8em; 
        color: rgba(255,255,255,0.5); 
        text-align: right; 
        margin-top: 5px; 
    }
    .action-buttons {
        display: flex;
        gap: 5px;
        margin-top: 5px;
    }
    .stButton button {
        padding: 2px 8px;
        font-size: 12px;
        height: auto;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

def safe_html(text: str) -> str:
    """Safely escape HTML characters"""
    return html.escape(str(text))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_bedrock_with_retry(client, **kwargs):
    """Invoke Bedrock API with retry logic"""
    try:
        return client.invoke_model(**kwargs)
    except Exception as e:
        if "ThrottlingException" in str(e):
            st.warning("Rate limit reached. Waiting before retry...")
            time.sleep(2)
        raise e

@st.cache_resource
def get_bedrock_client():
    """Initialize and cache the Bedrock client"""
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
    """Process and format a chat message"""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "reactions": {"likes": 0, "dislikes": 0},
        "thinking": thinking
    }

def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    """Get response from Claude via Bedrock"""
    try:
        with st.spinner("Thinking..."):
            response = invoke_bedrock_with_retry(
                client,
                modelId="anthropic.claude-v2",
                body=json.dumps({
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "anthropic_version": "bedrock-2023-05-31"
                })
            )
            
            response_body = json.loads(response['body'].read())
            full_response = response_body['completion']
            
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
            else:
                thinking = "Reasoning process not explicitly provided"
                main_response = full_response
                
            return thinking, main_response
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def display_message(message: dict, idx: int, current_chat: dict):
    """Display a chat message with actions"""
    if st.session_state.editing_message == idx and message["role"] == "user":
        edited_message = st.text_area("Edit message", message["content"], key=f"edit_{idx}")
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
        st.markdown(f"""<div class="message-content">{safe_html(message['content'])}</div>""", unsafe_allow_html=True)
        
        # Action buttons using Streamlit components
        cols = st.columns(4)
        with cols[0]:
            if st.button("📋 Copy", key=f"copy_{idx}"):
                st.write("Message copied to clipboard!")
                st.clipboard_copy(message["content"])
        
        if message["role"] == "user":
            with cols[1]:
                if st.button("✏️ Edit", key=f"edit_{idx}"):
                    st.session_state.editing_message = idx
                    st.rerun()
            with cols[2]:
                if st.button("🗑️ Delete", key=f"delete_{idx}"):
                    if st.session_state.current_chat in st.session_state.chats:
                        del st.session_state.chats[st.session_state.current_chat]["messages"][idx]
                        st.rerun()
        elif message["role"] == "assistant":
            with cols[1]:
                if st.button("🔄 Retry", key=f"retry_{idx}"):
                    retry_message(idx, current_chat)
            with cols[2]:
                if st.button(f"👍 {message['reactions'].get('likes', 0)}", key=f"like_{idx}"):
                    message['reactions']['likes'] = message['reactions'].get('likes', 0) + 1
                    st.rerun()
            with cols[3]:
                if st.button(f"👎 {message['reactions'].get('dislikes', 0)}", key=f"dislike_{idx}"):
                    message['reactions']['dislikes'] = message['reactions'].get('dislikes', 0) + 1
                    st.rerun()
        
        st.markdown(f"""<div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>""", unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and message.get('thinking'):
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(message['thinking'])

def retry_message(idx: int, current_chat: dict):
    """Retry generating a response for a specific message"""
    last_user_message = None
    for i in range(idx-1, -1, -1):
        if current_chat["messages"][i]["role"] == "user":
            last_user_message = current_chat["messages"][i]["content"]
            break
    
    if last_user_message:
        current_chat["messages"] = current_chat["messages"][:idx]
        client = get_bedrock_client()
        if client:
            thinking_process, main_response = get_chat_response(
                last_user_message,
                current_chat["messages"][-5:],
                client,
                current_chat["settings"]
            )
            if main_response:
                current_chat["messages"].append(
                    process_message(main_response, "assistant", thinking_process)
                )
                st.rerun()

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
    
    # Chat Management
    st.subheader("Chat Management")
    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create Chat") and new_chat_name:
        if new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "system_prompt": st.session_state.chats[st.session_state.current_chat]["system_prompt"],
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
    
    # Display Settings
    st.subheader("Display Settings")
    st.session_state.show_thinking = st.toggle(
        "Show Thinking Process", 
        value=st.session_state.show_thinking
    )
    
    # System Prompt
    st.subheader("System Prompt")
    current_chat["system_prompt"] = st.text_area(
        "System Prompt",
        value=current_chat["system_prompt"]
    )
    
    # Export and Clear options
    st.subheader("Chat Management")
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
        if st.button("Clear Chat"):
            if st.session_state.current_chat in st.session_state.chats:
                st.session_state.chats[st.session_state.current_chat]["messages"] = []
                st.rerun()

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Search functionality
st.session_state.search_query = st.text_input("🔍 Search messages")

# Display messages
messages_to_display = current_chat["messages"]
if st.session_state.search_query:
    search_term = st.session_state.search_query.lower()
    messages_to_display = [
        msg for msg in messages_to_display 
        if search_term in msg["content"].lower() 
        or search_term in (msg.get("thinking", "").lower())
    ]

for idx, message in enumerate(messages_to_display):
    with st.chat_message(message["role"]):
        display_message(message, idx, current_chat)

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    current_chat["messages"].append(process_message(prompt, "user"))
    
    # Get and display assistant response
    client = get_bedrock_client()
    if client:
        thinking_process, main_response = get_chat_response(
            prompt,
            current_chat["messages"][-5:],
            client,
            current_chat["settings"]
        )
        
        if main_response:
            current_chat["messages"].append(
                process_message(main_response, "assistant", thinking_process)
            )
            st.rerun()