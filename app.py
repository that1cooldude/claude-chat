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

# CSS and JavaScript with fixes
st.markdown("""
<style>
    .message-container { 
        margin: 15px 0; 
        padding: 15px; 
        border-radius: 10px; 
        position: relative; 
        border: 1px solid rgba(255,255,255,0.1);
        transition: box-shadow 0.3s ease;
    }
    .message-container:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .message-content {
        margin-bottom: 10px;
        white-space: pre-wrap;
        word-break: break-word;
        line-height: 1.5;
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
    .message-actions {
        position: absolute;
        right: 10px;
        top: 10px;
        opacity: 0;
        transition: opacity 0.2s ease;
        display: flex;
        gap: 8px;
        background: rgba(0,0,0,0.4);
        padding: 5px;
        border-radius: 5px;
    }
    .message-container:hover .message-actions { 
        opacity: 1; 
    }
    .action-btn { 
        padding: 5px 10px; 
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
        transform: translateY(-1px);
    }
    .reaction-btn.active { 
        background-color: rgba(50, 205, 50, 0.3); 
    }
    .favorite-prompt { 
        padding: 10px; 
        margin: 5px 0; 
        background-color: rgba(255,255,255,0.1); 
        border-radius: 5px; 
        cursor: pointer; 
    }
    .favorite-prompt:hover { 
        background-color: rgba(255,255,255,0.2); 
    }
    .edit-history {
        margin-top: 10px;
        padding: 10px;
        background: rgba(0,0,0,0.1);
        border-radius: 5px;
    }
    .edit-entry {
        margin: 5px 0;
        padding: 5px;
        border-left: 2px solid rgba(255,255,255,0.2);
    }
    .previous-content {
        font-size: 0.9em;
        color: rgba(255,255,255,0.7);
        margin-left: 10px;
    }
    .stForm {
        background: rgba(0,0,0,0.1);
        padding: 15px;
        border-radius: 10px;
    }
    
    @media (max-width: 768px) {
        .message-container { 
            margin: 10px 5px; 
            padding: 12px;
        }
        .message-actions { 
            opacity: 1;
            position: relative;
            right: auto;
            top: auto;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        .action-btn {
            flex: 1;
            justify-content: center;
        }
    }
</style>

<script>
function copyMessage(element) {
    const messageContainer = element.closest('.message-container');
    const messageContent = messageContainer.querySelector('.message-content');
    const text = messageContent.textContent.trim();
    
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    
    textarea.select();
    try {
        document.execCommand('copy');
        element.innerHTML = '✓ Copied!';
        element.style.backgroundColor = 'rgba(50, 205, 50, 0.3)';
    } catch (err) {
        element.innerHTML = '✗ Error!';
        element.style.backgroundColor = 'rgba(255, 0, 0, 0.3)';
    }
    
    document.body.removeChild(textarea);
    
    setTimeout(() => {
        element.innerHTML = '📋 Copy';
        element.style.backgroundColor = '';
    }, 2000);
}

function editMessage(idx) {
    window.streamlit.setComponentValue({
        action: 'edit',
        messageIdx: idx
    });
}

function deleteMessage(idx) {
    if (confirm('Delete this message?')) {
        window.streamlit.setComponentValue({
            action: 'delete',
            messageIdx: idx
        });
    }
}

function resendMessage(idx) {
    window.streamlit.setComponentValue({
        action: 'resend',
        messageIdx: idx
    });
}

function regenerateResponse(idx) {
    window.streamlit.setComponentValue({
        action: 'regenerate',
        messageIdx: idx
    });
}

function retryMessage(idx) {
    window.streamlit.setComponentValue({
        action: 'retry',
        messageIdx: idx
    });
}

function reactToMessage(idx, reaction) {
    window.streamlit.setComponentValue({
        action: 'react',
        messageIdx: idx,
        reaction: reaction
    });
}

function favoritePrompt(prompt) {
    window.streamlit.setComponentValue({
        action: 'favorite',
        prompt: prompt
    });
}
</script>
""", unsafe_allow_html=True)

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
if "favorite_prompts" not in st.session_state:
    st.session_state.favorite_prompts = []
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = {"active": False, "index": None, "content": None}

def safe_html(text: str) -> str:
    """Safely escape HTML characters"""
    return html.escape(str(text))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_bedrock_with_retry(client, **kwargs):
    """Invoke Bedrock with retry logic"""
    try:
        return client.invoke_model(**kwargs)
    except Exception as e:
        if "ThrottlingException" in str(e):
            st.warning("Rate limit reached. Waiting before retry...")
            time.sleep(2)
        raise e

@st.cache_resource
def get_bedrock_client():
    """Initialize and cache Bedrock client"""
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
    """Process and format a message"""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "reactions": {"likes": 0, "dislikes": 0},
        "thinking": thinking
    }

def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    """Get response from Claude"""
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

def export_chat_to_csv(chat):
    """Export chat to CSV format"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking Process', 'Timestamp', 'Reactions'])
    for message in chat["messages"]:
        writer.writerow([
            message["role"],
            message["content"],
            message.get("thinking", ""),
            message.get("timestamp", ""),
            json.dumps(message.get("reactions", {}))
        ])
    return output.getvalue()

def get_message_actions(idx: int, message: dict) -> str:
    """Generate message action buttons HTML"""
    actions = []
    if message["role"] == "user":
        actions.extend([
            f'<button class="action-btn" onclick="editMessage({idx})">✏️ Edit</button>',
            f'<button class="action-btn" onclick="resendMessage({idx})">🔄 Resend</button>',
            f'<button class="action-btn" onclick="deleteMessage({idx})">🗑️ Delete</button>'
        ])
    elif message["role"] == "assistant":
        actions.extend([
            f'<button class="action-btn" onclick="regenerateResponse({idx})">🔄 Regenerate</button>',
            f'<button class="action-btn reaction-btn" onclick="reactToMessage({idx}, \'like\')">👍 {message.get("reactions", {}).get("likes", 0)}</button>',
            f'<button class="action-btn reaction-btn" onclick="reactToMessage({idx}, \'dislike\')">👎 {message.get("reactions", {}).get("dislikes", 0)}</button>'
        ])
    return "\n".join(actions)

def get_edit_history(idx: int) -> str:
    """Generate edit history HTML for a message"""
    history = [h for h in st.session_state.message_history if h["original_idx"] == idx]
    if not history:
        return ""
    
    history_html = ['<div class="edit-history">']
    for entry in history:
        history_html.append(f"""
            <div class="edit-entry">
                <small>Previous version ({entry["timestamp"]}):</small>
                <div class="previous-content">{safe_html(entry["content"])}</div>
            </div>
        """)
    history_html.append('</div>')
    return "\n".join(history_html)

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
    
    # Export Options
    st.subheader("Export Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export JSON"):
            st.download_button(
                "Download JSON",
                data=json.dumps(current_chat, indent=2),
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("Export CSV"):
            csv_data = export_chat_to_csv(current_chat)
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    # Clear Chat
    if st.button("Clear Chat"):
        if st.session_state.current_chat in st.session_state.chats:
            st.session_state.chats[st.session_state.current_chat]["messages"] = []
            st.rerun()

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Search functionality
st.session_state.search_query = st.text_input("🔍 Search messages", key="search")

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
        if st.session_state.edit_mode["active"] and st.session_state.edit_mode["index"] == idx:
            # Create a form for editing
            with st.form(key=f"edit_form_{idx}"):
                edited_content = st.text_area(
                    "Edit message",
                    value=message["content"],
                    key=f"edit_area_{idx}"
                )
                col1, col2 = st.columns([1,1])
                with col1:
                    submit = st.form_submit_button("Save")
                    if submit:
                        # Store original message in history
                        st.session_state.message_history.append({
                            "original_idx": idx,
                            "content": message["content"],
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        # Update message
                        current_chat["messages"][idx]["content"] = edited_content
                        st.session_state.edit_mode = {"active": False, "index": None, "content": None}
                        st.rerun()
                with col2:
                    if st.form_submit_button("Cancel"):
                        st.session_state.edit_mode = {"active": False, "index": None, "content": None}
                        st.rerun()
        else:
            # Normal message display
            st.markdown(f"""
            <div class="message-container {message['role']}-message">
                <div class="message-content">{safe_html(message['content'])}</div>
                <div class="message-actions">
                    <button class="action-btn" onclick="copyMessage(this)">📋 Copy</button>
                    {get_message_actions(idx, message)}
                </div>
                <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
                {get_edit_history(idx) if message["role"] == "user" else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Display thinking process for assistant messages
            if message['role'] == 'assistant' and message.get('thinking'):
                with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                    st.markdown(f"""
                    <div class="thinking-container">
                        <div class="message-content">{safe_html(message['thinking'])}</div>
                        <button class="action-btn" onclick="copyMessage(this)">📋 Copy</button>
                    </div>
                    """, unsafe_allow_html=True)

# Handle message actions
if st.session_state.get('message_action'):
    action = st.session_state.message_action
    if action['action'] == 'edit':
        st.session_state.edit_mode = {
            "active": True,
            "index": action['messageIdx'],
            "content": current_chat["messages"][action['messageIdx']]["content"]
        }
    elif action['action'] == 'delete':
        del current_chat["messages"][action['messageIdx']]
    elif action['action'] == 'resend':
        # Get message content and resend
        message_content = current_chat["messages"][action['messageIdx']]["content"]
        current_chat["messages"].append(process_message(message_content, "user"))
        client = get_bedrock_client()
        if client:
            thinking_process, main_response = get_chat_response(
                message_content,
                current_chat["messages"][-5:],
                client,
                current_chat["settings"]
            )
            if main_response:
                current_chat["messages"].append(
                    process_message(main_response, "assistant", thinking_process)
                )
    elif action['action'] == 'regenerate':
        # Find last user message and regenerate response
        last_user_message = None
        for i in range(action['messageIdx']-1, -1, -1):
            if current_chat["messages"][i]["role"] == "user":
                last_user_message = current_chat["messages"][i]["content"]
                break
        if last_user_message:
            current_chat["messages"] = current_chat["messages"][:action['messageIdx']]
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
    elif action['action'] == 'react':
        msg_idx = action['messageIdx']
        reaction = action['reaction']
        if 'reactions' not in current_chat["messages"][msg_idx]:
            current_chat["messages"][msg_idx]['reactions'] = {'likes': 0, 'dislikes': 0}
        current_chat["messages"][msg_idx]['reactions'][f"{reaction}s"] += 1
    elif action['action'] == 'favorite':
        if action['prompt'] not in st.session_state.favorite_prompts:
            st.session_state.favorite_prompts.append(action['prompt'])
    
    st.session_state.message_action = None
    st.rerun()

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
