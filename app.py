import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
from time import sleep

# ... (keep existing imports)

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
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide"
)

# ... (keep existing session state initialization)

# -----------------------------------------------------------------------------
# Auto-refresh with optimized rerun
# -----------------------------------------------------------------------------
def auto_refresh():
    while True:
        sleep(1)  # Refresh every second
        st.experimental_rerun()

# Start auto-refresh in a background thread
threading.Thread(target=auto_refresh, daemon=True).start()

# -----------------------------------------------------------------------------
# Typing indicator
# -----------------------------------------------------------------------------
def show_typing_indicator():
    st.markdown(
        """
        <div style="
            color: #666;
            font-style: italic;
            text-align: center;
            padding: 1rem 0;
        ">
            Claude is typing...
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Chat UI Improvements
# -----------------------------------------------------------------------------
def build_message_ui(msg, show_thinking):
    if msg["role"] == "user":
        align = "flex-end"
        bubble_color = "#2e3136"
    else:
        align = "flex-start"
        bubble_color = "#454a50"
    
    st.markdown(
        f"""
        <div style="
            display: flex;
            {align};
            gap: 1rem;
            margin-top: 1rem;
        ">
            <div style="
                background-color: {bubble_color};
                color: white;
                padding: 12px 16px;
                border-radius: 10px;
                max-width: 80%;
                word-wrap: break-word;
            ">
                {msg['content']}
            </div>
            <div style="
                font-size: 0.75rem;
                color: rgba(255,255,255,0.5);
            ">
                {msg.get('timestamp', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if msg["role"] == "assistant" and show_thinking and msg.get("thinking"):
        with st.expander("Chain-of-Thought"):
            st.markdown(
                f"""
                <div style="
                    background-color: #333;
                    border-left: 3px solid #ffd700;
                    border-radius: 5px;
                    padding: 8px;
                    margin-top: 4px;
                ">
                    <span style="
                        color: #ffd700;
                        font-style: italic;
                        white-space: pre-wrap;
                        word-break: break-word;
                    ">
                        {msg['thinking']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------------------------------------------------------
# Main Layout
# -----------------------------------------------------------------------------
chat_col, settings_col = st.columns([2,1], gap="large")

# Chat Column
with chat_col:
    st.header("Claude Chat")
    
    # Create a container for the chat display
    chat_container = st.container()
    
    with chat_container:
        # Display messages with new UI
        for msg in get_current_chat_data()["messages"]:
            build_message_ui(msg, st.session_state.show_thinking)
        
        # Add typing indicator
        if st.session_state.get("typing", False):
            show_typing_indicator()
    
    # Create a form for message input and send button
    with st.form("message_form"):
        st.subheader("Type Your Message")
        st.session_state.user_input
