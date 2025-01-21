import streamlit as st
import boto3
from bedrock.runtime import BedrockRuntimeClient
import json
from datetime import datetime
import re
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
from time import sleep

# CONFIG
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-west-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-west-2::model(us.anthropic.claude-3-sonnet-20241022-v2)"
)

st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize AWS Bedrock runtime client
def init_bedrock_runtime():
    try:
        return BedrockRuntimeClient()
    except Exception as e:
        st.error(f"Failed to initialize AWS Bedrock runtime client: {e}")
        st.stop()

bedrock_runtime = init_bedrock_runtime()

# Session state initialization
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
    if "new_messages_since_last_update" not in st.session_state:
        st.session_state.new_messages_since_last_update = False
    if "typing" not in st.session_state:
        st.session_state.typing = False

# Auto-refresh with optimized rerun
def auto_refresh():
    while True:
        sleep(1)  # Refresh every second
        try:
            st.experimental_rerun()
        except Exception as e:
            print(f"Error in auto-refresh: {e}")

# Start auto-refresh in a background thread
threading.Thread(target=auto_refresh, daemon=True).start()

# Typing indicator
def show_typing_indicator():
    st.markdown(
        """
        <div style="color: #666; font-style: italic; text-align: center; padding: 1rem 0;">
            Claude is typing...
        </div>
        """,
        unsafe_allow_html=True
    )

# Chat UI Improvements
def build_message_ui(msg, show_thinking):
    if msg["role"] == "user":
        align = "flex-end"
        bubble_color = "#2e3136"
    else:
        align = "flex-start"
        bubble_color = "#454a50"
    
    st.markdown(
        f"""
        <div style="display: flex; {align}; gap: 1rem; margin-top: 1rem;">
            <div style="background-color: {bubble_color}; color: white; padding: 12px 16px; border-radius: 10px; max-width: 80%; word-wrap: break-word;">
                {msg['content']}
            </div>
            <div style="font-size: 0.75rem; color: rgba(255,255,255,0.5);">
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
                <div style="background-color: #333; border-left: 3px solid #ffd700; border-radius: 5px; padding: 8px; margin-top: 4px;">
                    <span style="color: #ffd700; font-style: italic; white-space: pre-wrap; word-break: break-word;">
                        {msg['thinking']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

# Function to converse with Claude
def converse_with_claude(prompt):
    try:
        response = bedrock_runtime.invoke_model(
            model_arn=MODEL_ARN,
            body=json.dumps({
                "messages": [
                    {
                        "role": "system",
                        "content": get_current_chat_data()["system_prompt"]
                    },
                    {
                        "role": "assistant",
                        "content": "Ok, I'm ready to help."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "top_p": 0.9
            }),
            accept="application/json",
            content_type="application/json"
        )

        response_body = json.loads(response.read())
        return {
            "role": "assistant",
            "content": response_body["messages"][-1]["content"],
            "thinking": response_body.get("thoughts", "")
        }

    except ClientError as e:
        st.error(f"AWS Bedrock API error: {e}")
        return {
            "role": "assistant",
            "content": "Sorry, I encountered an error while processing your request. Please try again.",
            "thinking": ""
        }

# Function to get current chat data
def get_current_chat_data():
    current_chat = st.session_state.current_chat
    if current_chat not in st.session_state.chats:
        st.session_state.current_chat = "Default"
        current_chat = "Default"
    return st.session_state.chats.get(
        current_chat, {"messages": []}
    )

# Main Layout
def main():
    init_session()
    
    # Use columns for better layout
    chat_col, settings_col = st.columns([3, 1])

    # Chat Column
    with chat_col:
        st.header("Claude Chat")
        
        # Chat messages container
        chat_container = st.container()
        with chat_container:
            current_chat_data = get_current_chat_data()
            if isinstance(current_chat_data.get('messages'), list):
                try:
                    for msg in current_chat_data["messages"]:
                        build_message_ui(msg, st.session_state.show_thinking)
                except Exception as e:
                    st.error(f"Error in message display: {e}")
            
            # Show typing indicator
            if st.session_state.get("typing", False):
                show_typing_indicator()
        
        # Form for sending messages
        with st.form("message_form", clear_on_submit=True):
            st.subheader("Type Your Message")
            user_input = st.text_area(
                "",
                value="",
                height=100,
                key="user_input_text_form",
                placeholder="Type your message here..."
            )
            
            # Controls
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.checkbox(
                    "Show Chain-of-Thought",
                    key="show_thinking",
                    value=st.session_state.show_thinking
                )
            
            with col2:
                st.checkbox(
                    "Force Chain-of-Thought",
                    key="force_thinking",
                    value=(get_current_chat_data()["force_thinking"] if "force_thinking" in get_current_chat_data() else False)
                )
            
            with col3:
                st.empty()  # Placeholder for alignment

            # Send button
            if st.form_submit_button("Send"):
                prompt = user_input.strip()
                if prompt:
                    try:
                        # Add user message
                        current_chat_data = get_current_chat_data()
                        if isinstance(current_chat_data, dict):
                            current_chat_data["messages"].append({
                                "role": "user",
                                "content": prompt,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            st.session_state.new_messages_since_last_update = True

                            # Get Claude's response
                            assistant_response = converse_with_claude(prompt)
                            if assistant_response:
                                current_chat_data["messages"].append(assistant_response)
                    except Exception as e:
                        st.error(f"Error sending message: {e}")

if __name__ == "__main__":
    main()
