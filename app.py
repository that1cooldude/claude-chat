import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------------------------------------------------------
# CONFIG - Adjust for your environment or store in st.secrets
# -----------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Claude Chat (Minimal)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------
# Minimal CSS
# -----------------------------------------------------------------------
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# State Initialization
# -----------------------------------------------------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_text" not in st.session_state:
        st.session_state.kb_text = ""
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False

init_session()

# -----------------------------------------------------------------------
# AWS Clients
# -----------------------------------------------------------------------
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to init Bedrock: {e}")
        return None

@st.cache_resource
def get_s3_client():
    try:
        return boto3.client(
            service_name="s3",
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to init S3: {e}")
        return None

# -----------------------------------------------------------------------
# S3 Save/Load
# -----------------------------------------------------------------------
def save_chat(chat_name, messages):
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
        st.success(f"Saved to s3://{S3_BUCKET_NAME}/{key}")
    except ClientError as e:
        st.error(f"S3 save error: {e}")

def load_chat(chat_name):
    s3 = get_s3_client()
    if not s3:
        return None
    key = f"conversations/{chat_name}.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        text = resp["Body"].read().decode("utf-8")
        data = json.loads(text)
        return data.get("messages", [])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        st.error(f"S3 load error: {e}")
        return None

# -----------------------------------------------------------------------
# Knowledge Base
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Claude Chat
# -----------------------------------------------------------------------
def build_messages(messages, kb_text):
    """
    Build the entire conversation for Claude. 
    For minimal usage, we won't use system messages at all 
    because 'system' role can cause validation errors on this endpoint.
    We just do 'user' and 'assistant'.
    If we have KB text, we inject it as a user message at the start.
    """
    bedrock_msgs = []
    # If we have knowledge base text, treat it as a user message:
    if kb_text.strip():
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Knowledge:\n" + kb_text}
            ]
        })
    # Then add user+assistant messages
    for msg in messages:
        bedrock_msgs.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })
    return bedrock_msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def get_claude_response(client, messages, kb_text, max_tokens=1000, temperature=0.7):
    with st.spinner("Claude is thinking..."):
        bedrock_msgs = build_messages(messages, kb_text)
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
        if "content" in body:
            combined_text = "\n".join(seg["text"] for seg in body["content"] if seg["type"]=="text")
            # Extract <thinking> if present
            thinking = ""
            match = re.search(r"<thinking>(.*?)</thinking>", combined_text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                combined_text = re.sub(r"<thinking>.*?</thinking>", "", combined_text, flags=re.DOTALL).strip()
            return combined_text, thinking
        return "No response content from Claude.", ""

def approximate_tokens(text):
    return len(text.split())

# -----------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------
st.title("Claude Chat (Minimal)")

chat_name = st.text_input("Chat Name to Save/Load", value="Default", key="chat_name_input")
col1, col2 = st.columns(2)
with col1:
    if st.button("Save Chat"):
        save_chat(chat_name, st.session_state.messages)
with col2:
    if st.button("Load Chat"):
        loaded_msgs = load_chat(chat_name)
        if loaded_msgs is None:
            st.warning("No chat found.")
        else:
            st.session_state.messages = loaded_msgs
            st.success("Loaded chat from S3")
            st.rerun()

# Knowledge Base
kb_input = st.text_input("KB S3 Key (e.g. 'mykb/somefile.txt')", value="")
if st.button("Load KB"):
    kb_data = load_kb_from_s3(kb_input)
    if kb_data:
        st.session_state.kb_text = kb_data
        st.success("Knowledge Base loaded.")
    else:
        st.session_state.kb_text = ""

# Show Thinking
st.session_state.show_thinking = st.checkbox("Show Thinking", value=st.session_state.show_thinking)

# Model Settings
temp = st.slider("Temperature", 0.0, 1.0, 0.7)
maxtok = st.slider("Max Tokens", 100, 4096, 1000)

# Token Usage
tok_sum = sum(approximate_tokens(msg["content"]) for msg in st.session_state.messages)
st.write(f"Approx. Tokens in conversation: {tok_sum}")

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        st.caption(msg.get("timestamp",""))
        # If assistant message has hidden thinking, show if toggled
        if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
            st.markdown(f"<div class='thinking-box'>{msg['thinking']}</div>", unsafe_allow_html=True)

prompt = st.chat_input("Ask Claude")
if prompt:
    # Add user message
    st.session_state.messages.append({
        "role":"user",
        "content": prompt,
        "thinking": "",
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    # Get Claude response
    client = get_bedrock_client()
    if client:
        ans, think = get_claude_response(
            client, st.session_state.messages,
            st.session_state.kb_text,
            max_tokens=maxtok,
            temperature=temp
        )
        st.session_state.messages.append({
            "role":"assistant",
            "content": ans,
            "thinking": think,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
    st.rerun()