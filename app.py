import streamlit as st 
import boto3
import json
import os
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

# Clean, professional CSS with better mobile support
st.markdown("""
<style>
    .stChat message {
        background-color: #2e3136;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .message-timestamp {
        font-size: 0.8em;
        color: rgba(255,255,255,0.5);
        margin-top: 5px;
    }
    @media (max-width: 768px) {
        .stButton button {
            padding: 0.5rem !important;
            width: auto !important;
        }
        .row-widget.stButton {
            margin: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# Session State Initialization
# ----------------------------------------------------------------------------------

def init_session_state():
    """Ensure session state has default structures."""
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
    
    # Ensure the current_chat key actually exists
    if st.session_state.current_chat not in st.session_state.chats:
        st.session_state.current_chat = "Default"
        if "Default" not in st.session_state.chats:
            st.session_state.chats["Default"] = {
                "messages": [],
                "settings": {"temperature": 0.7, "max_tokens": 1000},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

init_session_state()

# ----------------------------------------------------------------------------------
# AWS Bedrock / Claude Integration
# ----------------------------------------------------------------------------------

@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock client with error handling."""
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

def build_bedrock_messages(conversation):
    """
    Convert the entire conversation (list of dicts) into
    the JSON structure required by Claude on Bedrock.
    """
    bedrock_msgs = []
    for msg in conversation:
        bedrock_msgs.append({
            "role": msg["role"],  
            "content": [
                {
                    "type": "text",
                    "text": msg["content"]
                }
            ]
        })
    return bedrock_msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
def get_chat_response(client, conversation, settings):
    """
    Get response from Claude, using the entire conversation for multi-turn context.
    """
    try:
        with st.spinner("Claude is thinking..."):
            payload_messages = build_bedrock_messages(conversation)

            response = client.invoke_model(
                modelId=(
                    "arn:aws:bedrock:us-east-2:127214158930:inference-profile/"
                    "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                ),
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "top_k": 250,
                    "top_p": 0.999,
                    "messages": payload_messages
                })
            )
            
            response_body = json.loads(response.get('body').read())
            if 'content' in response_body:
                collected_text = []
                for item in response_body['content']:
                    if item.get('type') == 'text':
                        collected_text.append(item['text'])
                final_answer = "\n".join(collected_text).strip()
                return final_answer or "No response generated. Please try again."
            else:
                return "No response content found in Claude's answer."
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None

# ----------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------

def process_message(message: str, role: str) -> dict:
    """Wrap message text along with a timestamp and role."""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }

def save_chat_to_folder(chat_name, chat_data):
    """
    Saves a chat (conversation) to a 'conversations' folder in JSON format.
    """
    os.makedirs("conversations", exist_ok=True)
    file_path = os.path.join("conversations", f"{chat_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)
    st.success(f"Conversation '{chat_name}' saved to {file_path}.")

def load_chat_from_folder(chat_name):
    """
    Loads a chat from `conversations/{chat_name}.json`, if it exists.
    Returns None if the file is not found.
    """
    file_path = os.path.join("conversations", f"{chat_name}.json")
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------

with st.sidebar:
    st.title("Chat Settings")

    try:
        # Chat Management
        st.subheader("Conversations")
        new_chat_name = st.text_input("New Chat Name", key="new_chat_input")
        if st.button("Create Chat"):
            if new_chat_name and new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "settings": {"temperature": 0.7, "max_tokens": 1000},
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()

        # Select existing conversation
        st.session_state.current_chat = st.selectbox(
            "Select Chat",
            options=list(st.session_state.chats.keys()),
            index=list(st.session_state.chats.keys()).index(st.session_state.current_chat)
            if st.session_state.current_chat in st.session_state.chats
            else 0,
            key="chat_selector"
        )
        
        # Model Settings
        st.subheader("Model Settings")
        current_chat = st.session_state.chats[st.session_state.current_chat]

        current_chat["settings"]["temperature"] = st.slider(
            "Temperature",
            0.0, 1.0,
            current_chat["settings"].get("temperature", 0.7),
            help="Higher = more creative / explorative. Lower = more focused / deterministic."
        )
        
        current_chat["settings"]["max_tokens"] = st.slider(
            "Max Tokens",
            100, 4096,
            current_chat["settings"].get("max_tokens", 1000),
            help="Maximum length of Claude's responses."
        )
        
        # Save and Load
        st.subheader("Storage")
        col_save, col_load = st.columns(2)
        with col_save:
            if st.button("Save Chat"):
                save_chat_to_folder(st.session_state.current_chat, current_chat)

        with col_load:
            if st.button("Load Chat"):
                loaded = load_chat_from_folder(st.session_state.current_chat)
                if loaded is not None:
                    st.session_state.chats[st.session_state.current_chat] = loaded
                    st.success(f"Conversation '{st.session_state.current_chat}' loaded from folder.")
                    st.rerun()
                else:
                    st.warning(f"No saved file found for '{st.session_state.current_chat}'.")

        # Clear Chat with confirmation
        st.subheader("Clear Conversation")
        if st.button("Clear Current Chat"):
            if len(current_chat["messages"]) > 0:
                if st.button("Confirm Clear?"):
                    current_chat["messages"] = []
                    st.rerun()
            else:
                current_chat["messages"] = []
                st.rerun()

    except Exception as e:
        st.error(f"Error in sidebar: {str(e)}")
        init_session_state()

# ----------------------------------------------------------------------------------
# Main Chat Interface
# ----------------------------------------------------------------------------------

st.title(f"Claude Chat - {st.session_state.current_chat}")

try:
    # Display existing messages
    for idx, message in enumerate(current_chat["messages"]):
        with st.chat_message(message["role"]):
            if st.session_state.editing_message == idx and message["role"] == "user":
                # If user is editing a message, show text_area
                edited_message = st.text_area(
                    "Edit your message",
                    message["content"],
                    key=f"edit_{idx}"
                )
                col_save, col_cancel = st.columns([1, 1])
                with col_save:
                    if st.button("Save", key=f"save_{idx}"):
                        current_chat["messages"][idx]["content"] = edited_message
                        st.session_state.editing_message = None
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{idx}"):
                        st.session_state.editing_message = None
                        st.rerun()
            else:
                # Normal display
                st.markdown(message["content"])
                st.caption(f"Time: {message['timestamp']}")

                # For user messages, show Edit / Delete
                if message["role"] == "user":
                    c1, c2, _ = st.columns([1,1,8])
                    with c1:
                        if st.button("✏️", key=f"editbtn_{idx}", help="Edit message"):
                            st.session_state.editing_message = idx
                            st.rerun()
                    with c2:
                        if st.button("🗑️", key=f"delbtn_{idx}", help="Delete message"):
                            current_chat["messages"].pop(idx)
                            st.rerun()

    # Chat input
    prompt = st.chat_input("Message Claude...", key="chat_input")
    if prompt:
        # Add user's new message to conversation
        current_chat["messages"].append(process_message(prompt, "user"))

        # Get Claude response using entire conversation
        client = get_bedrock_client()
        if client:
            response = get_chat_response(
                client=client,
                conversation=current_chat["messages"],  # pass entire conversation
                settings=current_chat["settings"]
            )
            if response:
                current_chat["messages"].append(
                    process_message(response, "assistant")
                )
        st.experimental_rerun()

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.button("Reset Application"):
        init_session_state()
        st.rerun()