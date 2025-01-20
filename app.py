import streamlit as st
import boto3
import json
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO
import csv
import html

# Page configuration
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .message-container {
        margin: 15px 0;
        padding: 15px;
        border-radius: 10px;
        position: relative;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .user-message { background-color: #2e3136; margin-left: 20px; }
    .assistant-message { background-color: #36393f; margin-right: 20px; }
    .timestamp {
        font-size: 0.8em;
        color: rgba(255,255,255,0.5);
        text-align: right;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": "Structure EVERY response with thinking and final answer sections.",
            "settings": {"temperature": 0.7, "max_tokens": 1000},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default"

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
        st.error(f"Error initializing Bedrock client: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_bedrock_with_retry(client, model_id, payload):
    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload)
        )
        return json.loads(response['body'].read())
    except Exception as e:
        st.error(f"Error invoking Bedrock model: {str(e)}")
        raise

def process_message(message, role, thinking=None):
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "thinking": thinking
    }

def export_chat_to_csv(chat):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking Process', 'Timestamp'])
    for msg in chat["messages"]:
        writer.writerow([
            msg["role"], msg["content"], msg.get("thinking", ""), msg["timestamp"]
        ])
    return output.getvalue()

# Sidebar
with st.sidebar:
    st.title("Claude Chat Settings")
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
            st.experimental_rerun()

    st.session_state.current_chat = st.selectbox(
        "Select Chat",
        options=list(st.session_state.chats.keys())
    )

    # Model settings
    current_chat = st.session_state.chats[st.session_state.current_chat]
    current_chat["settings"]["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, current_chat["settings"]["temperature"]
    )
    current_chat["settings"]["max_tokens"] = st.slider(
        "Max Tokens", 100, 4096, current_chat["settings"]["max_tokens"]
    )

# Main Chat UI
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")
current_chat = st.session_state.chats[st.session_state.current_chat]

if prompt := st.chat_input("Message Claude..."):
    client = get_bedrock_client()
    if client:
        try:
            # Call Claude model
            payload = {
                "prompt": f"Human: {prompt}\n\nAssistant:",
                "temperature": current_chat["settings"]["temperature"],
                "max_tokens_to_sample": current_chat["settings"]["max_tokens"]
            }
            response = invoke_bedrock_with_retry(client, "anthropic.claude-v2", payload)
            thinking, answer = response.get("thinking", ""), response.get("completion", "")
            
            # Save messages
            current_chat["messages"].append(process_message(prompt, "user"))
            current_chat["messages"].append(process_message(answer, "assistant", thinking))
        except Exception as e:
            st.error(f"Failed to process message: {str(e)}")

# Display messages
for message in current_chat["messages"]:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f"""
    <div class="message-container {role_class}">
        <div>{html.escape(message["content"])}</div>
        <div class="timestamp">{message["timestamp"]}</div>
    </div>
    """, unsafe_allow_html=True)