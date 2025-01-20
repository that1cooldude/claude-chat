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

# CSS and JavaScript
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
    .reaction-btn.active { background-color: rgba(50, 205, 50, 0.3); }
    .favorite-prompt {
        padding: 10px;
        margin: 5px 0;
        background-color: rgba(255,255,255,0.1);
        border-radius: 5px;
        cursor: pointer;
    }
    .favorite-prompt:hover { background-color: rgba(255,255,255,0.2); }
    .edit-area {
        margin-top: 10px;
        background-color: rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 5px;
    }
    @media (max-width: 768px) {
        .message-container { margin: 10px 5px; }
        .message-actions {
            opacity: 1;
            position: relative;
            right: auto;
            top: auto;
            margin-top: 10px;
        }
        .stButton>button { width: 100%; }
    }
    .stChatInputContainer textarea {
        background-color: #2e3136;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        padding: 10px;
    }
    .stChatInputContainer textarea:focus {
        border-color: #6495ED;
        outline: none;
    }
    .stChatInputContainer button {
        background-color: #6495ED;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        cursor: pointer;
    }
    .stChatInputContainer button:hover {
        background-color: #558ED5;
    }
</style>

<script>
function copyMessage(element) {
    const messageContainer = element.closest('.message-container');
    const messageContent = messageContainer.querySelector('.message-content');
    const text = messageContent.textContent.trim();
    navigator.clipboard.writeText(text).then(() => {
        element.innerText = 'Copied!';
        element.style.backgroundColor = 'rgba(50, 205, 50, 0.3)';
        setTimeout(() => {
            element.innerText = 'Copy';
            element.style.backgroundColor = '';
        }, 2000);
    }).catch(() => {
        element.innerText = 'Error!';
        setTimeout(() => element.innerText = 'Copy', 2000);
    });
}

function editMessage(idx) {
    window.streamlitApp.setComponentValue({action: 'edit', messageIdx: idx});
}

function deleteMessage(idx) {
    if (confirm('Delete this message?')) {
        window.streamlitApp.setComponentValue({action: 'delete', messageIdx: idx});
    }
}

function retryMessage(idx) {
    window.streamlitApp.setComponentValue({action: 'retry', messageIdx: idx});
}

function reactToMessage(idx, reaction) {
    window.streamlitApp.setComponentValue({action: 'react', messageIdx: idx, reaction: reaction});
}

function favoritePrompt(prompt) {
    window.streamlitApp.setComponentValue({action: 'favorite', prompt: prompt});
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
if "message_action" not in st.session_state:
    st.session_state.message_action = None
if "favorite_prompts" not in st.session_state:
    st.session_state.favorite_prompts = []
if "reactions" not in st.session_state:
    st.session_state.reactions = {}

def safe_html(text: str) -> str:
    return html.escape(str(text))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_bedrock_with_retry(client, **kwargs):
    try:
        return client.invoke_model(**kwargs)
    except Exception as e:
        if "ThrottlingException" in str(e):
            st.warning("Rate limit reached. Waiting before retry...")
            time.sleep(2)
        raise e

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

def process_message(message: str, role: str, thinking: str = None, tool_calls=None) -> dict:
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "reactions": {"likes": 0, "dislikes": 0},
        "thinking": thinking,
        "tool_calls": tool_calls
    }

def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    try:
        with st.spinner("Thinking..."):
            response = invoke_bedrock_with_retry(
                client,
                modelId="anthropic.claude-v2",
                body=json.dumps({
                    "prompt": f"{''.join([f'\\n\\n{msg['role'].capitalize()}: {msg['content']}' for msg in conversation_history])}\\n\\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "anthropic_version": "bedrock-2023-05-31"
                })
            )

            response_body = json.loads(response['body'].read())
            full_response = response_body['completion']

            # Basic tool call detection (you'll need to adjust this based on the model's output format)
            tool_calls = []
            if "<tool_code>" in full_response and "</tool_code>" in full_response:
                tool_match = re.search(r'<tool_code>(.*?)</tool_code>', full_response, re.DOTALL)
                if tool_match:
                    try:
                        tool_calls = json.loads(tool_match.group(1).strip())
                        full_response = re.sub(r'<tool_code>.*?</tool_code>', '', full_response, flags=re.DOTALL).strip()
                    except json.JSONDecodeError:
                        print("Could not decode tool call JSON")

            # Split thinking and response
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
            else:
                thinking = "Reasoning process not explicitly provided"
                main_response = full_response

            return thinking, main_response, tool_calls

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None

def execute_tool_call(tool_call):
    """
    This is a placeholder function for executing tool calls.
    You'll need to implement the logic for each tool.
    """
    print(f"Executing tool: {tool_call}")
    time.sleep(2) # Simulate tool execution
    return f"Tool '{tool_call['name']}' executed successfully with parameters: {tool_call['parameters']}"

def export_chat_to_csv(chat):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking Process', 'Timestamp', 'Reactions', 'Tool Calls'])
    for message in chat["messages"]:
        writer.writerow([
            message["role"],
            message["content"],
            message.get("thinking", ""),
            message.get("timestamp", ""),
            json.dumps(message.get("reactions", {})),
            json.dumps(message.get("tool_calls", []))
        ])
    return output.getvalue()

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
        if st.session_state.editing_message == idx and message["role"] == "user":
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
            # Prepare action buttons
            action_buttons = ""
            if message["role"] == "user":
                action_buttons = f"""
                <button class="action-btn" onclick="editMessage({idx})">Edit</button>
                <button class="action-btn" onclick="deleteMessage({idx})">Delete</button>
                """
                if message["content"] not in st.session_state.favorite_prompts:
                    action_buttons += f"""
                    <button class="action-btn" onclick="favoritePrompt('{safe_html(message['content'])}')">
                        Favorite
                    </button>
                    """
            elif message["role"] == "assistant":
                action_buttons = f"""
                <button class="action-btn" onclick="retryMessage({idx})">Retry</button>
                <button class="action-btn" onclick="reactToMessage({idx}, 'like')">
                    👍 {message.get('reactions', {}).get('likes', 0)}
                </button>
                <button class="action-btn" onclick="reactToMessage({idx}, 'dislike')">
                    👎 {message.get('reactions', {}).get('dislikes', 0)}
                </button>
                """

            # Display message
            st.markdown(f"""
            <div class="message-container {message['role']}-message">
                <div class="message-content">{safe_html(message['content'])}</div>
                <div class="message-actions">
                    <button class="action-btn" onclick="copyMessage(this)">Copy</button>
                    {action_buttons}
                </div>
                <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Display thinking process for assistant messages
            if message['role'] == 'assistant' and message.get('thinking'):
                with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                    st.markdown(f"""
                    <div class="thinking-container">
                        <div class="message-content">{safe_html(message['thinking'])}</div>
                        <button class="action-btn" onclick="copyMessage(this)">Copy</button>
                    </div>
                    """, unsafe_allow_html=True)

            # Display tool calls
            if message['role'] == 'assistant' and message.get('tool_calls'):
                with st.expander("Tool Calls", expanded=False):
                    for tool_call in message['tool_calls']:
                        st.write(f"Calling tool: `{tool_call['name']}`")
                        st.json(tool_call['parameters'])
                        # Here you would ideally display the result of the tool call if available

# Handle message actions
if 'message_action' in st.session_state and st.session_state.message_action:
    action = st.session_state.message_action
    if action['action'] == 'edit':
        st.session_state.editing_message = action['messageIdx']
    elif action['action'] == 'delete':
        del current_chat["messages"][action['messageIdx']]
    elif action['action'] == 'retry':
        last_user_message = None
        messages_before = current_chat["messages"][:action['messageIdx']]
        for msg in reversed(messages_before):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        if last_user_message:
            current_chat["messages"] = messages_before
            client = get_bedrock_client()
            if client:
                thinking_process, main_response, tool_calls = get_chat_response(
                    last_user_message,
                    current_chat["messages"],
                    client,
                    current_chat["settings"]
                )
                if main_response:
                    current_chat["messages"].append(
                        process_message(main_response, "assistant", thinking_process, tool_calls)
                    )
                    if tool_calls:
                        # Basic handling: Execute tool calls and add the results as a new message
                        for tool_call in tool_calls:
                            tool_result = execute_tool_call(tool_call)
                            current_chat["messages"].append(
                                process_message(tool_result, "tool_response", tool_name=tool_call['name'])
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

    # Display user message
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            <div class="message-content">{safe_html(prompt)}</div>
            <div class="message-actions">
                <button class="action-btn" onclick="copyMessage(this)">Copy</button>
            </div>
            <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Get and display assistant response
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        if client:
            thinking_process, main_response, tool_calls = get_chat_response(
                prompt,
                current_chat["messages"],
                client,
                current_chat["settings"]
            )

            if main_response:
                message_obj = process_message(main_response, "assistant", thinking_process, tool_calls)
                current_chat["messages"].append(message_obj)

                st.markdown(f"""
                <div class="message-container assistant-message">
                    <div class="message-content">{safe_html(main_response)}</div>
                    <div class="message-actions">
                        <button class="action-btn" onclick="copyMessage(this)">Copy</button>
                        <button class="action-btn" onclick="retryMessage({len(current_chat['messages'])-1})">Retry</button>
                        <button class="action-btn" onclick="reactToMessage({len(current_chat['messages'])-1}, 'like')">👍 0</button>
                        <button class="action-btn" onclick="reactToMessage({len(current_chat['messages'])-1}, 'dislike')">👎 0</button>
                    </div>
                    <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
                </div>
                """, unsafe_allow_html=True)

                if thinking_process:
                    with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                        st.markdown(f"""
                        <div class="thinking-container">
                            <div class="message-content">{safe_html(thinking_process)}</div>
                            <button class="action-btn" onclick="copyMessage(this)">Copy</button>
                        </div>
                        """, unsafe_allow_html=True)

                if tool_calls:
                    with st.expander("Tool Calls", expanded=False):
                        for tool_call in tool_calls:
                            st.write(f"Calling tool: `{tool_call['name']}`")
                            st.json(tool_call['parameters'])
                            # Here you would trigger the tool execution and display the result
                            # For simplicity, we'll just print a placeholder message
                            st.info(f"Executing tool '{tool_call['name']}'...")
                            tool_result = execute_tool_call(tool_call) # Execute the tool
                            current_chat["messages"].append(process_message(tool_result, "tool_response", tool_name=tool_call['name']))
                            st.write(f"Tool '{tool_call['name']}' response:")
                            st.write(tool_result)
