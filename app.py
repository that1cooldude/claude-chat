import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# --------------------------------------------------------------------------------
# CONFIG - Fill in or use st.secrets
# --------------------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Claude Chat (No System Role)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# CSS for simple styling
# --------------------------------------------------------------------------------
st.markdown("""
<style>
    .thinking-box {
        background-color: #1e1e2e; 
        border-left: 3px solid #ffd700; 
        padding: 10px; 
        margin-top: 10px; 
        font-style: italic;
        border-radius: 5px;
    }
    .chat-bubble {
        margin: 15px 0;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .assistant-bubble {
        background-color: #36393f;
    }
    .user-bubble {
        background-color: #2e3136;
    }
    .msg-timestamp {
        font-size: 0.8em;
        color: rgba(255,255,255,0.5);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        # Each item: { "role": "user"/"assistant", "content": "...", "thinking": "...", "timestamp": "..." }
        st.session_state.messages = []
    if "kb_text" not in st.session_state:
        st.session_state.kb_text = ""
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
    if "editing_idx" not in st.session_state:
        st.session_state.editing_idx = None

init_session_state()

# --------------------------------------------------------------------------------
# AWS Clients
# --------------------------------------------------------------------------------
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
        st.error(f"Bedrock client error: {e}")
        return None

@st.cache_resource
def get_s3_client():
    try:
        return boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"S3 client error: {e}")
        return None

# --------------------------------------------------------------------------------
# S3 Save/Load
# --------------------------------------------------------------------------------
def save_chat_to_s3(chat_name, messages):
    s3 = get_s3_client()
    if not s3:
        return
    data = {"messages": messages}
    key = f"conversations/{chat_name}.json"
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType="application/json"
        )
        st.success(f"Chat '{chat_name}' saved to s3://{S3_BUCKET_NAME}/{key}")
    except ClientError as e:
        st.error(f"Save error: {e}")

def load_chat_from_s3(chat_name):
    s3 = get_s3_client()
    if not s3:
        return None
    key = f"conversations/{chat_name}.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        raw = resp["Body"].read().decode("utf-8")
        data = json.loads(raw)
        return data.get("messages", [])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        st.error(f"Load error: {e}")
        return None

# --------------------------------------------------------------------------------
# Knowledge Base
# --------------------------------------------------------------------------------
def load_kb_from_s3(kb_key):
    s3 = get_s3_client()
    if not s3:
        return ""
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=kb_key)
        return resp["Body"].read().decode("utf-8")
    except ClientError as e:
        st.error(f"KB load error: {e}")
        return ""

# --------------------------------------------------------------------------------
# Claude Chat Logic
# --------------------------------------------------------------------------------
def build_bedrock_messages(messages, kb_text):
    """
    We only use 'user' and 'assistant' roles. 
    If kb_text is present, we prepend a user message with that knowledge.
    """
    bedrock_msgs = []

    # Insert knowledge as if user said it (so it doesn't break the "role" schema).
    if kb_text.strip():
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Knowledge Base:\n{kb_text}"}
            ]
        })

    # Then the real conversation
    for msg in messages:
        bedrock_msgs.append({
            "role": msg["role"],  # "user" or "assistant"
            "content": [{"type": "text", "text": msg["content"]}]
        })
    return bedrock_msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, messages, kb_text, temperature, max_tokens):
    with st.spinner("Claude is thinking..."):
        bedrock_msgs = build_bedrock_messages(messages, kb_text)
        resp = client.invoke_model(
            modelId=MODEL_ARN,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": 250,
                "top_p": 0.999,
                "messages": bedrock_msgs
            })
        )
        body = json.loads(resp["body"].read())
        # Merge text
        if "content" in body:
            combined = "\n".join(seg["text"] for seg in body["content"] if seg["type"]=="text")
            # Extract <thinking>
            thinking = ""
            match = re.search(r"<thinking>(.*?)</thinking>", combined, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                combined = re.sub(r"<thinking>.*?</thinking>", "", combined, flags=re.DOTALL).strip()
            return combined, thinking
        return "No response content.", ""

def approximate_tokens(text):
    return len(text.split())

def total_token_usage(messages):
    return sum(approximate_tokens(m["content"]) for m in messages)

def create_message(role, content, thinking=""):
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# --------------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------------
st.title("Claude Chat - No System Role")

chat_name = st.text_input("Current Chat Name", value="Default", key="chat_name_box")
colA, colB = st.columns(2)
with colA:
    if st.button("Save Chat"):
        save_chat_to_s3(chat_name, st.session_state.messages)
with colB:
    if st.button("Load Chat"):
        loaded = load_chat_from_s3(chat_name)
        if loaded is None:
            st.warning("No chat found for that name.")
        else:
            st.session_state.messages = loaded
            st.success(f"Loaded '{chat_name}' from S3.")
            st.rerun()

# Knowledge Base
kb_key = st.text_input("Knowledge Base S3 Key", value="", key="kb_key_box")
if st.button("Load KB"):
    text = load_kb_from_s3(kb_key)
    if text:
        st.session_state.kb_text = text
        st.success("KB loaded.")
    else:
        st.session_state.kb_text = ""

# Show Thinking
st.session_state.show_thinking = st.checkbox("Show Thinking", value=st.session_state.show_thinking)

# Model Settings
temp = st.slider("Temperature", 0.0, 1.0, 0.7)
maxtok = st.slider("Max Tokens", 100, 4096, 1000)

# Token Usage
tk_usage = total_token_usage(st.session_state.messages)
st.write(f"Approx. Token Usage: **{tk_usage}**")

# Clear Chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# --------------------------------------------------------------------------------
# MAIN CHAT DISPLAY
# --------------------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    # Distinguish user vs assistant style
    style_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"

    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and st.session_state.editing_idx == i:
            edited_text = st.text_area("Edit your message", msg["content"], key=f"edit_{i}")
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Save", key=f"save_{i}"):
                    st.session_state.messages[i]["content"] = edited_text
                    st.session_state.editing_idx = None
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"cancel_{i}"):
                    st.session_state.editing_idx = None
                    st.rerun()
        else:
            # Normal message display
            st.markdown(f"<div class='chat-bubble {style_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            st.caption(f"{msg['timestamp']}")

            # If assistant, show chain-of-thought
            if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
                st.markdown(f"<div class='thinking-box'>{msg['thinking']}</div>", unsafe_allow_html=True)

            # For user messages, an Edit / Delete row
            if msg["role"]=="user":
                col1, col2, _ = st.columns([1,1,8])
                with col1:
                    if st.button("✏️ Edit", key=f"editbtn_{i}"):
                        st.session_state.editing_idx = i
                        st.rerun()
                with col2:
                    if st.button("🗑️ Del", key=f"delbtn_{i}"):
                        st.session_state.messages.pop(i)
                        st.rerun()

# --------------------------------------------------------------------------------
# BOTTOM INPUT
# --------------------------------------------------------------------------------
user_input = st.chat_input("Message Claude...")
if user_input:
    # Add user message
    st.session_state.messages.append(create_message("user", user_input))
    # Get Claude response
    client = get_bedrock_client()
    if client:
        reply, thinking = invoke_claude(client, st.session_state.messages, st.session_state.kb_text, temp, maxtok)
        st.session_state.messages.append(create_message("assistant", reply, thinking))
    st.rerun()