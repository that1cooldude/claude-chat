import streamlit as st
from st_autoreload import st_autoreload  # Add this import
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Derek's Claude Chat",
    page_icon="🤖",
    layout="wide"
)

# Add auto-reload configuration
st_autoreload(interval=1)  # Auto-refresh every second

# -----------------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------------
def init_session():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Default": {
                "messages": [],
                "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                "force_thinking": False
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""
    if "new_messages" not in st.session_state:
        st.session_state.new_messages = False

init_session()

# -----------------------------------------------------------------------------
# AWS CLIENTS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Error creating Bedrock client: {e}")
        return None

# -----------------------------------------------------------------------------
# UI IMPROVEMENTS
# -----------------------------------------------------------------------------
def update_ui():
    st.session_state.new_messages = True

# Enhanced styles
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
    max-height: 80vh;
    overflow-y: auto;
}
.user-bubble {
    background-color: #2e3136;
    color: #fff;
    padding: 12px 16px;
    border-radius: 10px;
    max-width: 80%;
    align-self: flex-end;
    word-wrap: break-word;
    margin-bottom: 4px;
    font-size: 16px;
}
.assistant-bubble {
    background-color: #454a50;
    color: #fff;
    padding: 12px 16px;
    border-radius: 10px;
    max-width: 80%;
    align-self: flex-start;
    word-wrap: break-word;
    margin-bottom: 4px;
    font-size: 16px;
    border-left: 3px solid #00ff9d;
}
.timestamp {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-top: 2px;
    text-align: right;
}
.thinking-expander {
    background-color: #333;
    border-left: 3px solid #ffd700;
    border-radius: 5px;
    padding: 8px;
    margin-top: 4px;
}
.thinking-text {
    color: #ffd700;
    font-style: italic;
    white-space: pre-wrap;
    word-break: break-word;
}
.loading-spinner {
    text-align: center;
    padding: 2rem;
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
with st.expander("Conversation Settings", expanded=True):
    chat_keys = list(st.session_state.chats.keys())
    chosen_chat = st.selectbox(
        "Switch Conversation 🔄",
        options=chat_keys,
        index=chat_keys.index(st.session_state.current_chat)
        if st.session_state.current_chat in chat_keys
        else 0
    )
    if chosen_chat != st.session_state.current_chat:
        st.session_state.current_chat = chosen_chat
        update_ui()

col_chat, col_settings = st.columns([2,1], gap="medium")

with col Chat:
    st.header(f"💬 Derek's Claude Chat")
    st.write("Type your message below and let Claude respond. Use the settings on the right to customize your experience.")

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    chat_data = get_current_chat_data()
    for msg in chat_data["messages"]:
        bubble_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='timestamp'>{msg.get('timestamp','')}</div>", unsafe_allow_html=True)

        if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
            with st.expander("Chain-of-Thought ⚙️"):
                st.markdown(
                    f"<div class='thinking-expander'><span class='thinking-text'>{msg['thinking']}</span></div>",
                    unsafe_allow_html=True
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # Loading state
    if st.session_state.new_messages:
        with st.spinner("Generating response..."):
            pass

    # Message input
    st.subheader("Type Your Message 💬")
    st.session_state.user_input_text = st.text_area(
        "Message",
        value=st.session_state.user_input_text,
        height=100
    )

    if st.button("Send Message"):
        user_msg_str = st.session_state.user_input_text.strip()
        if user_msg_str:
            chat_data["messages"].append({
                "role": "user",
                "content": user_msg_str,
                "thinking": "",
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            
            client = get_bedrock_client()
            if client:
                ans_text, ans_think = invoke_claude(client, chat_data, temperature, max_tokens)
                chat_data["messages"].append({
                    "role": "assistant",
                    "content": ans_text,
                    "thinking": ans_think,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                update_ui()  # Trigger UI refresh

            st.session_state.user_input_text = ""
        else:
            st.warning("Please enter a message first")

with col_settings:
    st.header("Settings 🛠️")
    
    # System Prompt
    st.subheader("System Prompt")
    chat_data["system_prompt"] = st.text_area(
        "Claude's instructions",
        value=chat_data.get("system_prompt",""),
        height=100
    )

    # Model Controls
    st.subheader("Model Controls")
    temperature = st.slider("Temperature 💧", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens 📏", 100, 4096, 1000)

    # Thinking options
    st.subheader("Chain-of-Thought 🤔")
    chat_data["force_thinking"] = st.checkbox(
        "Force chain-of-thought in <thinking> tags",
        value=chat_data.get("force_thinking",False)
    )
    st.session_state.show_thinking = st.checkbox(
        "Show internal reasoning",
        value=st.session_state.show_thinking
    )

    # Conversation management
    st.subheader("Manage Conversations")
    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create New Chat"):
        if new_chat_name and new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                "force_thinking": False
            }
            st.session_state.current_chat = new_chat_name
            update_ui()

    if st.button("Clear Current Chat"):
        chat_data["messages"] = []
        update_ui()

    # Usage
    st.subheader("Stats 📊")
    tok_count = total_token_usage(chat_data)
    st.write(f"Approximate tokens used: **{tok_count}**")

    # Save/Load to S3
    st.subheader("Cloud Storage 🌁")
    save_status = st.button("Save to S3")
    if save_status:
        save_chat_to_s3(st.session_state.current_chat, chat_data)
        update_ui()

    loaded_data = None
    load_status = st.button("Load from S3")
    if load_status:
        loaded_data = load_chat_from_s3(st.session_state.current_chat)
        if loaded_data is not None:
            st.session_state.chats[st.session_state.current_chat] = loaded_data
            st.session_state.current_chat = st.session_state.current_chat
            update_ui()

    # Auto-refresh status
    st.subheader("Auto-Refresh ✅")
    st.write("The app automatically refreshes every second when new messages are available.")
