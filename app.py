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
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved CSS for better UI
st.markdown("""
<style>
    /* Message Container */
    .message-container {
        margin: 15px 0;
        padding: 15px;
        border-radius: 10px;
        position: relative;
        border: 1px solid rgba(255,255,255,0.1);
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
    
    /* Message Content */
    .message-content {
        margin-bottom: 10px;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* Copy Button */
    .copy-button {
        position: absolute;
        top: 5px;
        right: 5px;
        padding: 5px 10px;
        background: rgba(255,255,255,0.1);
        border: none;
        border-radius: 4px;
        color: white;
        cursor: pointer;
        font-size: 12px;
        transition: all 0.3s ease;
    }
    .copy-button:hover {
        background: rgba(255,255,255,0.2);
    }
    .copy-button.copied {
        background: #4CAF50;
    }
    
    /* Timestamps and Additional UI Elements */
    .timestamp {
        font-size: 0.8em;
        color: rgba(255,255,255,0.5);
        position: absolute;
        bottom: 5px;
        right: 10px;
    }
    .thinking-container {
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 10px;
        margin: 10px 0;
        font-style: italic;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .message-container {
            margin: 10px 5px;
            padding: 10px;
        }
        .copy-button {
            opacity: 1;
            position: static;
            margin-top: 10px;
            width: 100%;
        }
    }
</style>

<script>
// Improved copy functionality with proper error handling
function copyMessage(messageId) {
    const messageElement = document.getElementById(messageId);
    const textToCopy = messageElement.innerText;
    const buttonElement = messageElement.parentElement.querySelector('.copy-button');
    
    navigator.clipboard.writeText(textToCopy)
        .then(() => {
            buttonElement.textContent = '✓ Copied!';
            buttonElement.classList.add('copied');
            setTimeout(() => {
                buttonElement.textContent = 'Copy';
                buttonElement.classList.remove('copied');
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            buttonElement.textContent = '✗ Error';
            setTimeout(() => {
                buttonElement.textContent = 'Copy';
            }, 2000);
        });
}
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "temperature": 0.7,
        "max_tokens": 1000
    }

# Bedrock client setup
@st.cache_resource
def get_bedrock_client():
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

# Message processing with retry
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    try:
        with st.spinner("Thinking..."):
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
            return response_body['completion'].strip()
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None

# Sidebar for settings
with st.sidebar:
    st.title("Chat Settings")
    st.session_state.settings["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    st.session_state.settings["max_tokens"] = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100
    )
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
st.title("🤖 Claude Chat")

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    message_id = f"message_{idx}"
    with st.chat_message(message["role"]):
        st.markdown(f"""
            <div class="message-container {message['role']}-message">
                <div class="message-content" id="{message_id}">{html.escape(message['content'])}</div>
                <button class="copy-button" onclick="copyMessage('{message_id}')">Copy</button>
                <div class="timestamp">{message.get('timestamp', datetime.now().strftime('%I:%M %p'))}</div>
            </div>
        """, unsafe_allow_html=True)

# Chat input and response handling
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime('%I:%M %p')
    })
    
    # Get and display assistant response
    client = get_bedrock_client()
    if client:
        response = get_chat_response(
            prompt,
            st.session_state.            st.session_state.chat_history[-5:],
            client,
            st.session_state.settings
        )
        
        if response:
            # Add assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime('%I:%M %p')
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime('%I:%M %p')
            })
            st.rerun()

# Export functionality
def export_chat():
    chat_data = {
        "messages": st.session_state.messages,
        "settings": st.session_state.settings,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # JSON export
    json_str = json.dumps(chat_data, indent=2)
    st.download_button(
        label="Download Chat (JSON)",
        data=json_str,
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )
    
    # CSV export
    csv_data = StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(['Role', 'Content', 'Timestamp'])
    
    for message in st.session_state.messages:
        csv_writer.writerow([
            message['role'],
            message['content'],
            message['timestamp']
        ])
    
    st.download_button(
        label="Download Chat (CSV)",
        data=csv_data.getvalue(),
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Add export buttons to sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("Export Chat")
    export_chat()

# Error handling
def handle_error(error_message: str):
    st.error(f"Error: {error_message}")
    with st.expander("Error Details"):
        st.code(str(error_message))

# Optional: Add keyboard shortcuts
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.keyCode === 13) {
        document.querySelector('.stButton button').click();
    }
});
</script>
""", unsafe_allow_html=True)

# Optional: Add system status
with st.sidebar:
    st.markdown("---")
    st.subheader("System Status")
    if client:
        st.success("Connected to Claude")
    else:
        st.error("Not connected to Claude")
    
    st.markdown(f"Messages: {len(st.session_state.messages)}")
    st.markdown(f"Temperature: {st.session_state.settings['temperature']}")
    st.markdown(f"Max Tokens: {st.session_state.settings['max_tokens']}")

# Optional: Add message search
search_query = st.sidebar.text_input("Search messages...")
if search_query:
    filtered_messages = [
        msg for msg in st.session_state.messages
        if search_query.lower() in msg['content'].lower()
    ]
    st.sidebar.markdown(f"Found {len(filtered_messages)} matches")
    for msg in filtered_messages:
        st.sidebar.markdown(f"""
        **{msg['role']}** ({msg['timestamp']}):
        >{msg['content'][:100]}...
        """)

# Add welcome message if chat is empty
if not st.session_state.messages:
    st.markdown("""
    👋 Welcome to Claude Chat! 
    
    Start by typing a message below. You can:
    - Use Ctrl/Cmd + Enter to send messages
    - Search through messages using the sidebar
    - Export your chat history as JSON or CSV
    - Adjust temperature and max tokens in settings
    
    Happy chatting! 🚀
    """)

