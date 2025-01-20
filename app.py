import streamlit as st
import boto3
import json
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import csv
from io import StringIO

# Page Configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS with fixed message display
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    .message-container {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .user-message {
        background-color: rgba(70, 70, 70, 0.2);
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: rgba(50, 50, 50, 0.2);
        margin-right: 2rem;
    }
    
    .message-content {
        white-space: pre-wrap;
        word-break: break-word;
        margin-right: 100px;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .message-actions {
        position: absolute;
        right: 1rem;
        top: 1rem;
        display: flex;
        gap: 0.5rem;
    }
    
    .action-button {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.2s;
    }
    
    .action-button:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .action-button.copied {
        background-color: #4CAF50;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        text-align: right;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .thinking-container {
        background-color: rgba(30, 30, 46, 0.5);
        border-left: 3px solid #ffd700;
        padding: 1rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    @media (max-width: 768px) {
        .message-actions {
            position: relative;
            right: auto;
            top: auto;
            opacity: 1;
            margin-top: 0.5rem;
        }
        .message-content {
            margin-right: 0;
        }
    }
    
    .stButton > button {
        width: auto;
    }
    
    #MainMenu, footer, header {display: none;}
    .stDeployButton {display: none;}
</style>

<script>
function copyMessage(element, messageId) {
    const container = document.querySelector(`#message-${messageId}`);
    const content = container.querySelector('.message-content').textContent;
    
    navigator.clipboard.writeText(content).then(() => {
        element.textContent = 'Copied!';
        element.classList.add('copied');
        setTimeout(() => {
            element.textContent = 'Copy';
            element.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        element.textContent = 'Error!';
        setTimeout(() => element.textContent = 'Copy', 2000);
    });
}

function editMessage(messageId) {
    window.streamlit.setComponentValue({
        type: 'edit_message',
        messageId: messageId
    });
}

function deleteMessage(messageId) {
    if (confirm('Delete this message?')) {
        window.streamlit.setComponentValue({
            type: 'delete_message',
            messageId: messageId
        });
    }
}

function retryMessage(messageId) {
    window.streamlit.setComponentValue({
        type: 'retry_message',
        messageId: messageId
    });
}
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 1000
if 'show_thinking' not in st.session_state:
    st.session_state.show_thinking = True

# AWS Bedrock client setup - ORIGINAL VERSION
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-2'
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

def safe_html(text):
    """Escape HTML special characters"""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ai_response(prompt: str, temperature: float, max_tokens: int) -> tuple[str, str]:
    client = get_bedrock_client()
    if not client:
        return None, "Failed to initialize AI client"
    
    try:
        response = client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        
        response_body = json.loads(response['body'].read())
        full_response = response_body['completion']
        
        thinking = ""
        main_response = full_response
        
        thinking_start = full_response.find('<thinking>')
        thinking_end = full_response.find('</thinking>')
        
        if thinking_start != -1 and thinking_end != -1:
            thinking = full_response[thinking_start + 10:thinking_end].strip()
            main_response = full_response[thinking_end + 11:].strip()
        
        return thinking, main_response
        
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return None, str(e)

def add_message(role: str, content: str, thinking: str = None):
    st.session_state.messages.append({
        'role': role,
        'content': content,
        'thinking': thinking,
        'timestamp': datetime.now().strftime('%I:%M %p')
    })

def export_chat():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking', 'Timestamp'])
    
    for msg in st.session_state.messages:
        writer.writerow([
            msg['role'],
            msg['content'],
            msg.get('thinking', ''),
            msg['timestamp']
        ])
    
    return output.getvalue()

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1
    )
    
    st.session_state.max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=st.session_state.max_tokens,
        step=100
    )
    
    st.session_state.show_thinking = st.toggle(
        "Show Thinking Process",
        value=st.session_state.show_thinking
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Export Chat"):
        csv_data = export_chat()
        st.download_button(
            label="Download Chat",
            data=csv_data,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# Main chat interface
st.title("💭 AI Chat Assistant")

# Display messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Only display the actual message content, no UI elements
        message_content = message["content"].strip()
        
        st.markdown(f"""
        <div class="message-container {message['role']}-message" id="message-{idx}">
            <div class="message-content">{safe_html(message_content)}</div>
            <div class="message-actions">
                <button class="action-button" onclick="copyMessage(this, {idx})">Copy</button>
                {"<button class='action-button' onclick='editMessage(" + str(idx) + ")'>Edit</button>" if message['role'] == 'user' else ""}
                {"<button class='action-button' onclick='deleteMessage(" + str(idx) + ")'>Delete</button>" if message['role'] == 'user' else ""}
                {"<button class='action-button' onclick='retryMessage(" + str(idx) + ")'>Retry</button>" if message['role'] == 'assistant' else ""}
            </div>
            <div class="timestamp">{message['timestamp']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and message.get('thinking'):
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(f"""
                <div class="thinking-container">
                    <div class="message-content">{safe_html(message['thinking'])}</div>
                </div>
                """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Message the AI..."):
    add_message('user', prompt)
    
    with st.spinner("AI is thinking..."):
        thinking, response = get_ai_response(
            prompt,
            st.session_state.temperature,
            st.session_state.max_tokens
        )
        
        if response:
            add_message('assistant', response, thinking)
            st.rerun()
        else:
            st.error("Failed to get AI response. Please try again.")

# Handle message actions
for event in st.session_state.get("_stcore_message_events", []):
    if event.get("type") == "edit_message":
        message_id = event["messageId"]
        new_content = st.text_input("Edit message", st.session_state.messages[message_id]["content"])
        if st.button("Save"):
            st.session_state.messages[message_id]["content"] = new_content
            st.rerun()
    
    elif event.get("type") == "delete_message":
        message_id = event["messageId"]
        st.session_state.messages.pop(message_id)
        st.rerun()
    
    elif event.get("type") == "retry_message":
        message_id = event["messageId"]
        for i in range(message_id - 1, -1, -1):
            if st.session_state.messages[i]["role"] == "user":
                prompt = st.session_state.messages[i]["content"]
                st.session_state.messages = st.session_state.messages[:message_id]
                thinking, response = get_ai_response(
                    prompt,
                    st.session_state.temperature,
                    st.session_state.max_tokens
                )
                if response:
                    add_message('assistant', response, thinking)
                break
        st.rerun()
