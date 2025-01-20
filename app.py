import streamlit as st
import boto3
import json
import os
import re
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

# ----------------------------------------------------------------------------------
# Basic CSS for improved mobile UI
# ----------------------------------------------------------------------------------
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
    .thinking-container {
        background-color: #1e1e2e; 
        border-left: 3px solid #ffd700; 
        padding: 10px; 
        margin-top: 10px; 
        font-style: italic;
        border-radius: 5px;
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
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "system_prompt": "Please include your chain-of-thought inside <thinking>...</thinking>."
                },
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False  # default off
    
    # Ensure the current_chat key actually exists
    if st.session_state.current_chat not in st.session_state.chats:
        st.session_state.current_chat = "Default"
        if "Default" not in st.session_state.chats:
            st.session_state.chats["Default"] = {
                "messages": [],
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "system_prompt": "Please include your chain-of-thought inside <thinking>...</thinking>."
                },
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

def build_bedrock_messages(conversation, system_prompt):
    """
    Convert the entire conversation (list of dicts) into
    the JSON structure required by Claude on Bedrock.
    We also insert a system message at the start (Claude's instructions).
    """
    bedrock_msgs = []

    # 1) Insert system message from user-defined system prompt
    bedrock_msgs.append({
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt
            }
        ]
    })

    # 2) Then add the user+assistant messages from conversation
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
    Get response from Claude, using entire conversation for multi-turn context.
    Attempt to parse out <thinking> tags if present.
    """
    try:
        with st.spinner("Claude is thinking..."):
            # Build the entire conversation
            system_prompt = settings.get("system_prompt", "")
            payload_messages = build_bedrock_messages(conversation, system_prompt)

            response = client.invoke_model(
                modelId=(
                    "arn:aws:bedrock:us-east-2:127214158930:"
                    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
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
                # Combine all returned text segments
                collected_text = []
                for item in response_body['content']:
                    if item.get('type') == 'text':
                        collected_text.append(item['text'])
                final_answer = "\n".join(collected_text).strip()
                
                # Attempt to extract <thinking> content
                thinking_match = re.search(r"<thinking>(.*?)</thinking>", final_answer, re.DOTALL)
                if thinking_match:
                    extracted_thinking = thinking_match.group(1).strip()
                    # Remove the thinking portion from the final visible text
                    visible_text = re.sub(r"<thinking>.*?</thinking>", "", final_answer, flags=re.DOTALL).strip()
                else:
                    extracted_thinking = None
                    visible_text = final_answer

                return visible_text, extracted_thinking or ""
            else:
                return "No response content found in Claude's answer.", ""

    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None, None

# ----------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------

def process_message(message: str, role: str, thinking: str = "") -> dict:
    """
    Wrap message text, thinking text, timestamp, and role.
    'thinking' is optional, only used for assistant messages.
    """
    return {
        "role": role,
        "content": message,
        "thinking": thinking,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }

def approximate_tokens(text: str) -> int:
    """Roughly count tokens by splitting on whitespace."""
    return len(text.split())

def total_token_usage(conversation) -> int:
    """Sum approximate tokens for all messages (excluding thinking)."""
    total = 0
    for msg in conversation:
        total += approximate_tokens(msg["content"])
    return total

def save_chat_to_folder(chat_name, chat_data):
    """Save entire chat to 'conversations' folder in JSON format."""
    os.makedirs("conversations", exist_ok=True)
    file_path = os.path.join("conversations", f"{chat_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)
    st.success(f"Conversation '{chat_name}' saved to {file_path}.")

def load_chat_from_folder(chat_name):
    """Load chat from folder if file exists, else return None."""
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
                    "settings": {
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "system_prompt": "Please include your chain-of-thought inside <thinking>...</thinking>."
                    },
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()

        st.session_state.current_chat = st.selectbox(
            "Select Chat",
            options=list(st.session_state.chats.keys()),
            index=list(st.session_state.chats.keys()).index(st.session_state.current_chat)
            if st.session_state.current_chat in st.session_state.chats
            else 0,
            key="chat_selector"
        )

        current_chat = st.session_state.chats[st.session_state.current_chat]

        # Model Settings
        st.subheader("Model Settings")
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
        # System Prompt
        st.subheader("System Prompt")
        current_chat["settings"]["system_prompt"] = st.text_area(
            "Claude will see this prompt before every conversation exchange, so you can shape behavior or style.",
            value=current_chat["settings"].get("system_prompt", ""),
        )

        # Show Thinking Toggle
        st.subheader("Thinking Process")
        st.session_state.show_thinking = st.checkbox(
            "Show Thinking",
            value=st.session_state.show_thinking,
            help="Display any <thinking> content parsed from Claude's response"
        )

        # Token Usage
        st.subheader("Approx. Token Usage")
        total_tokens = total_token_usage(current_chat["messages"])
        st.write(f"Total Conversation Tokens: **{total_tokens}**")

        # Internet Search using Perplexity
        st.subheader("Internet Search")
        perplex_query = st.text_input("Search Query", key="perplex_query")
        if st.button("Search Perplexity"):
            if perplex_query.strip():
                encoded_query = perplex_query.strip().replace(" ", "+")
                perplex_url = f"https://www.perplexity.ai/search?q={encoded_query}"
                st.markdown(
                    f"[Open Perplexity in new tab]({perplex_url})",
                    unsafe_allow_html=True
                )
            else:
                st.warning("Please enter a search query.")

        # Save / Load
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

        # Clear Chat
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
            if (message["role"] == "user") and (st.session_state.editing_message == idx):
                # Edit mode for user message
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

                # If assistant has a thinking chunk and user wants to show it
                if message["role"] == "assistant" and message.get("thinking") and st.session_state.show_thinking:
                    with st.expander("Thinking Process"):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {message["thinking"]}
                        </div>
                        """, unsafe_allow_html=True)

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
        # 1) Add user's new message to conversation
        current_chat["messages"].append(
            process_message(prompt, "user")
        )

        # 2) Get Claude response using entire conversation
        client = get_bedrock_client()
        if client:
            visible_answer, thinking_text = get_chat_response(
                client=client,
                conversation=current_chat["messages"],
                settings=current_chat["settings"]
            )
            if visible_answer is not None:
                # 3) Append assistant message
                current_chat["messages"].append(
                    process_message(visible_answer, "assistant", thinking=thinking_text)
                )

        st.experimental_rerun()

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.button("Reset Application"):
        init_session_state()
        st.rerun()