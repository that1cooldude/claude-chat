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

# Page configuration
st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Custom CSS and JavaScript for UI
st.markdown("""
<style>
    .message-container { 
        margin: 15px 0; 
        padding: 15px; 
        border-radius: 10px; 
        position: relative; 
        border: 1px solid rgba(255,255,255,0.1); 
    }
    .message-content {
        margin-bottom: 10px;
        white-space: pre-wrap;
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
    .message-actions {
        position: absolute;
        right: 5px;
        top: 5px;
        opacity: 0;
        transition: all 0.2s ease;
        display: flex;
        gap: 5px;
    }
    .message-container:hover .message-actions { opacity: 1; }
    .action-btn { 
        padding: 4px 8px; 
        background-color: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2); 
        border-radius: 4px; 
        color: #fff; 
        font-size: 12px; 
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .action-btn:hover { background-color: rgba(255,255,255,0.2); }
    @media (max-width: 768px) {
        .message-container { margin: 10px 5px; }
        .stButton>button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": "Structure EVERY response with thinking and final answer sections.",
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

def safe_html(text: str) -> str:
    """Escape HTML to prevent injection issues."""
    return html.escape(str(text))

@st.cache_resource
def get_bedrock_client():
    """Get a Bedrock client for AWS."""
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
    """Process message into a structured format."""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "thinking": thinking
    }

def export_chat_to_csv(chat):
    """Export chat to a CSV format."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking Process', 'Timestamp'])
    for message in chat["messages"]:
        writer.writerow([
            message["role"],
            message["content"],
            message.get("thinking", ""),
            message.get("timestamp", "")
        ])
    return output.getvalue()

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
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
            st.experimental_rerun()
    st.session_state.current_chat = st.selectbox("Select Chat", options=list(st.session_state.chats.keys()))

    current_chat = st.session_state.chats[st.session_state.current_chat]
    current_chat["settings"]["temperature"] = st.slider("Temperature", 0.0, 1.0, current_chat["settings"]["temperature"])
    current_chat["settings"]["max_tokens"] = st.slider("Max Tokens", 100, 4096, current_chat["settings"]["max_tokens"])
    st.session_state.show_thinking = st.checkbox("Show Thinking Process", value=st.session_state.show_thinking)

    if st.button("Clear Chat"):
        st.session_state.chats[st.session_state.current_chat]["messages"] = []
        st.experimental_rerun()

# Main Chat UI
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")
if st.chat_input("Message Claude..."):
    pass  # Implement Claude logic here