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
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved JavaScript with error handling
st.markdown("""
<script>
async function copyMessage(elementId) {
    try {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error('Element not found:', elementId);
            return;
        }
        
        const text = element.textContent || element.innerText;
        await navigator.clipboard.writeText(text);
        
        const button = element.parentElement.querySelector('.copy-button');
        if (button) {
            button.textContent = '✓ Copied!';
            button.classList.add('copy-success');
            setTimeout(() => {
                button.textContent = 'Copy';
                button.classList.remove('copy-success');
            }, 2000);
        }
    } catch (err) {
        console.error('Copy failed:', err);
        const button = element.parentElement.querySelector('.copy-button');
        if (button) {
            button.textContent = '✗ Error';
            button.classList.add('copy-error');
            setTimeout(() => {
                button.textContent = 'Copy';
                button.classList.remove('copy-error');
            }, 2000);
        }
    }
}

function handleMessageAction(action, messageId, data = {}) {
    try {
        const component = window.streamlitApp;
        if (!component) {
            console.error('Streamlit component not found');
            return;
        }
        
        component.setComponentValue({
            action: action,
            messageId: messageId,
            ...data
        });
    } catch (err) {
        console.error('Action failed:', err);
    }
}
</script>

<style>
    .message-container { 
        margin: 15px 0; 
        padding: 15px; 
        border-radius: 10px; 
        position: relative; 
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.2s ease;
    }
    .message-content {
        margin-bottom: 10px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .user-message { 
        background-color: #2e3136; 
        margin-left: 20px;
        border-left: 3px solid #4CAF50;
    }
    .assistant-message { 
        background-color: #36393f; 
        margin-right: 20px;
        border-left: 3px solid #2196F3;
    }
    .thinking-container { 
        background-color: #1e1e2e; 
        border-left: 3px solid #ffd700; 
        padding: 10px; 
        margin: 10px 0; 
        font-style: italic; 
    }
    .message-actions {
        position: absolute;
        right: 5px;
        top: 5px;
        opacity: 0;
        transition: opacity 0.2s ease;
        display: flex;
        gap: 5px;
    }
    .message-container:hover .message-actions { 
        opacity: 1; 
    }
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
    .action-btn:hover { 
        background-color: rgba(255,255,255,0.2);
    }
    .copy-success {
        background-color: #4CAF50 !important;
    }
    .copy-error {
        background-color: #f44336 !important;
    }
    
    @media (max-width: 768px) {
        .message-container { 
            margin: 10px 5px; 
        }
        .message-actions { 
            opacity: 1;
            position: relative;
            top: auto;
            right: auto;
            margin-top: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Improved session state initialization
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

# Helper functions with error handling
def safe_html(text: str) -> str:
    try:
        return html.escape(str(text))
    except Exception as e:
        st.error(f"HTML escape error: {str(e)}")
        return str(text)

def process_message(message: str, role: str, thinking: str = None) -> dict:
    try:
        return {
            "role": role,
            "content": message,
            "timestamp": datetime.now().strftime('%I:%M %p'),
            "reactions": {"likes": 0, "dislikes": 0},
            "thinking": thinking,
            "id": f"{role}_{int(time.time()*1000)}"
        }
    except Exception as e:
        st.error(f"Message processing error: {str(e)}")
        return {
            "role": role,
            "content": message,
            "reactions": {"likes": 0, "dislikes": 0},
            "id": f"{role}_{int(time.time()*1000)}"
        }

# AWS client with improved error handling
@st.cache_resource
def get_bedrock_client():
    try:
        client = boto3.client(
            service_name='bedrock-runtime',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        # Test connection
        client.list_custom_models()
        return client
    except Exception as e:
        st.error(f"AWS client error: {str(e)}")
        return None

# Message handling with retries
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    try:
        response = client.invoke_model(
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
        
        # Parse thinking section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
        else:
            thinking = "Reasoning process not explicitly provided"
            main_response = full_response
            
        return thinking, main_response
            
    except Exception as e:
        st.error(f"Chat response error: {str(e)}")
        return None, None

# Display message with error handling
def display_message(message: dict, idx: int):
    try:
        message_id = message.get('id', f"msg_{idx}")
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        
        st.markdown(f"""
        <div class="message-container {role_class}">
            <div class="message-content" id="{message_id}">{safe_html(message['content'])}</div>
            <div class="message-actions">
                <button class="action-btn" onclick="copyMessage('{message_id}')">Copy</button>
                {'<button class="action-btn" onclick="handleMessageAction(\'edit\', ' + str(idx) + ')">Edit</button>' if message["role"] == "user" else ''}
                {'<button class="action-btn" onclick="handleMessageAction(\'retry\', ' + str(idx) + ')">Retry</button>' if message["role"] == "assistant" else ''}
            </div>
            <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and message.get('thinking'):
            with st.expander("Thinking Process", expanded=st.session_state.get('show_thinking', True)):
                st.markdown(f"""
                <div class="thinking-container">
                    <div class="message-content">{safe_html(message['thinking'])}</div>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Display error: {str(e)}")

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
    
    current_chat = st.session_state.chats[st.session_state.current_chat]
    
    # Model settings
    st.subheader("Model Settings")
    current_chat["settings"]["temperature"] = st.slider(
        "Temperature",
        0.0, 1.0,
        current_chat["settings"]["temperature"]
    )
    
    # Clear chat
    if st.button("Clear Chat"):
        current_chat["messages"] = []
        st.rerun()

# Display messages
try:
    for idx, message in enumerate(current_chat["messages"]):
        with st.chat_message(message["role"]):
            display_message(message, idx)
except Exception as e:
    st.error(f"Message display error: {str(e)}")

# Chat input
if prompt := st.chat_input("Message Claude..."):
    try:
        # Add user message
        current_chat["messages"].append(process_message(prompt, "user"))
        
        # Get response
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
    except Exception as e:
        st.error(f"Chat processing error: {str(e)}")
