import streamlit as st
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
    page_title="Derek's Claude",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------------------------------------------
# STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
    max-height: 70vh;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 10px;
    background-color: #f8f8f8;
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
</style>
""", unsafe_allow_html=True)

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

    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""

init_session()

def get_current_chat_data():
    return st.session_state.chats[st.session_state.current_chat]

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
        st.error(f"Error creating S3 client: {e}")
        return None

# -----------------------------------------------------------------------------
# S3 Save/Load
# -----------------------------------------------------------------------------
def save_chat_to_s3(chat_name, chat_data):
    s3 = get_s3_client()
    if not s3:
        return
    key = f"conversations/{chat_name}.json"
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(chat_data, indent=2),
            ContentType="application/json"
        )
        st.success(f"Saved chat '{chat_name}' to s3://{S3_BUCKET_NAME}/{key}")
    except ClientError as e:
        st.error(f"Save error: {e}")

def load_chat_from_s3(chat_name):
    s3 = get_s3_client()
    if not s3:
        return None
    key = f"conversations/{chat_name}.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        st.error(f"Load error: {e}")
        return None

def list_chats_s3():
    s3 = get_s3_client()
    if not s3:
        return []
    try:
        objs = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="conversations/")
        if "Contents" not in objs:
            return []
        all_names = []
        for item in objs["Contents"]:
            k = item["Key"]
            if k.endswith(".json"):
                base = k.replace("conversations/","").replace(".json","")
                all_names.append(base)
        return sorted(all_names)
    except ClientError as e:
        st.error(f"List error: {e}")
        return []

# -----------------------------------------------------------------------------
# INVOKE CLAUDE
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, chat_data, temperature, max_tokens):
    bedrock_msgs = []

    if chat_data.get("force_thinking", False):
        bedrock_msgs.append({
            "role": "user",
            "content": [{"type": "text", "text": "Please include chain-of-thought in <thinking>...</thinking> if possible."}]
        })

    sys_prompt = chat_data.get("system_prompt", "").strip()
    if sys_prompt:
        bedrock_msgs.append({
            "role": "user",
            "content": [{"type": "text", "text": f"(System Prompt)\n{sys_prompt}"}]
        })

    for msg in chat_data["messages"]:
        bedrock_msgs.append({
            "role": msg["role"],
            "content": [
                {"type": "text", "text": msg["content"]}
            ]
        })

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
        text_bits = []
        for seg in body["content"]:
            if seg.get("type") == "text":
                text_bits.append(seg["text"])
        combined = "\n".join(text_bits).strip()

        thinking = ""
        match = re.search(r"<thinking>(.*?)</thinking>", combined, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            combined = re.sub(r"<thinking>.*?</thinking>", "", combined, flags=re.DOTALL).strip()
        return combined, thinking
    return "No response from Claude.", ""

def create_message(role, content, thinking=""):
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
col_chat, col_settings = st.columns([2, 1], gap="large")

with col_settings:
    st.title("Derek's Claude")

    chat_keys = list(st.session_state.chats.keys())
    chosen_chat = st.selectbox(
        "Pick a Conversation",
        options=chat_keys,
        index=chat_keys.index(st.session_state.current_chat)
        if st.session_state.current_chat in chat_keys
        else 0
    )
    if chosen_chat != st.session_state.current_chat:
        st.session_state.current_chat = chosen_chat

    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create Conversation"):
        if new_chat_name and new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                "force_thinking": False
            }
            st.session_state.current_chat = new_chat_name

    chat_data = get_current_chat_data()

    st.subheader("Model Controls")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1000)

    st.subheader("System Prompt")
    chat_data["system_prompt"] = st.text_area(
        "Claude instructions",
        value=chat_data.get("system_prompt", "")
    )

    st.subheader("Save and Load Chats")
    if st.button("Save Chat"):
        save_chat_to_s3(st.session_state.current_chat, chat_data)

    if st.button("Load Chat"):
        loaded_data = load_chat_from_s3(st.session_state.current_chat)
        if loaded_data is None:
            st.warning("No record in S3 for that name.")
        else:
            st.session_state.chats[st.session_state.current_chat] = loaded_data
            st.success("Loaded conversation from S3.")

with col_chat:
    st.header(f"Conversation: {st.session_state.current_chat}")

    chat_display = st.empty()

    with chat_display.container():
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for i, msg in enumerate(chat_data["messages"]):
            bubble_class = "assistant-bubble" if msg["role"] == "assistant" else "user-bubble"
            st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='timestamp'>{msg.get('timestamp', '')}</div>", unsafe_allow_html=True)

            if msg["role"] == "assistant" and msg.get("thinking"):
                with st.expander("Chain-of-Thought"):
                    st.markdown(
                        f"<div class='thinking-expander'><span class='thinking-text'>{msg['thinking']}</span></div>",
                        unsafe_allow_html=True
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Send a Message")
    user_input = st.text_area("Type your message", value=st.session_state.user_input_text, height=75, key="user_input")
    if st.button("Send Message"):
        if user_input.strip():
            chat_data["messages"].append({
                "role": "user",
                "content": user_input.strip(),
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

            st.session_state.user_input_text = ""
            st.success("Message sent and response received!")
        else:
            st.warning("Cannot send an empty message.")

    # Auto-refresh without blocking user interaction
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0

    st.session_state.refresh_counter += 1
    if st.session_state.refresh_counter % 100 == 0:
        st.experimental_rerun()
