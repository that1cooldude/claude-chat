import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# -------------------------------------------------------------------
# CONFIG (Use st.secrets or fill in your environment)
# -------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")

# Use the Claude inference profile of your choice:
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

# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# State Initialization
# -------------------------------------------------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
    if "force_thinking" not in st.session_state:
        st.session_state.force_thinking = False
    if "editing_idx" not in st.session_state:
        st.session_state.editing_idx = None

init_session_state()

# -------------------------------------------------------------------
# AWS Clients
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# S3 Save/Load
# -------------------------------------------------------------------
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
        st.success(f"Chat '{chat_name}' saved: s3://{S3_BUCKET_NAME}/{key}")
    except ClientError as e:
        st.error(f"Error saving chat: {e}")

def load_chat_from_s3(chat_name):
    s3 = get_s3_client()
    if not s3:
        return None
    key = f"conversations/{chat_name}.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        raw_data = resp["Body"].read().decode("utf-8")
        data = json.loads(raw_data)
        return data.get("messages", [])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        st.error(f"Error loading chat: {e}")
        return None

def list_saved_chats():
    """List existing chat files in s3://bucket/conversations/, returning base names."""
    s3 = get_s3_client()
    if not s3:
        return []
    try:
        objs = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="conversations/")
        if "Contents" not in objs:
            return []
        chat_names = []
        for item in objs["Contents"]:
            key = item["Key"]  # e.g. "conversations/SomeChat.json"
            if key.endswith(".json"):
                # strip prefix, strip extension
                base = key.replace("conversations/", "").replace(".json", "")
                chat_names.append(base)
        return sorted(chat_names)
    except ClientError as e:
        st.error(f"Error listing chats: {e}")
        return []

# -------------------------------------------------------------------
# Claude Logic
# -------------------------------------------------------------------
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

def build_bedrock_messages(messages, force_thinking):
    """
    We only use 'user' and 'assistant' roles. 
    If force_thinking is True, we prepend a user message instructing Claude to show chain-of-thought.
    """
    bedrock_msgs = []
    # If user toggled "force chain-of-thought," add an initial user message
    if force_thinking:
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type":"text","text": "You MUST show your chain-of-thought in <thinking>...</thinking> tags."}
            ]
        })

    # Then add the actual conversation
    for msg in messages:
        bedrock_msgs.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })
    return bedrock_msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, messages, force_thinking, temperature, max_tokens):
    """
    Multi-turn call to Claude with optional forced chain-of-thought. 
    We'll parse out <thinking> from the combined text.
    """
    with st.spinner("Claude is thinking..."):
        payload = build_bedrock_messages(messages, force_thinking)
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
                "messages": payload
            })
        )
        body = json.loads(resp["body"].read())
        if "content" in body:
            merged = "\n".join(seg["text"] for seg in body["content"] if seg["type"]=="text")
            # Extract <thinking> chunk
            thinking = ""
            match = re.search(r"<thinking>(.*?)</thinking>", merged, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                merged = re.sub(r"<thinking>.*?</thinking>", "", merged, flags=re.DOTALL).strip()
            return merged, thinking
        return ("No response returned by Claude.", "")

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.title("Claude Chat (No System Role)")

# 1) Toggle for chain-of-thought
st.session_state.force_thinking = st.checkbox(
    "Force Chain-of-Thought",
    value=st.session_state.force_thinking,
    help="If checked, we tell Claude it MUST show reasoning in <thinking> tags."
)

# 2) Show/Hide chain-of-thought
st.session_state.show_thinking = st.checkbox(
    "Show Thinking",
    value=st.session_state.show_thinking,
    help="If on, any extracted <thinking> text is displayed for assistant messages."
)

# 3) Model Settings
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.slider("Max Tokens", 100, 4096, 1000)

# 4) List saved chats from S3
saved_chats = list_saved_chats()
st.subheader("Load a Saved Chat")
if saved_chats:
    chosen_chat = st.selectbox(
        "Choose from existing S3 chat files:",
        options=saved_chats
    )
    if st.button("Load Selected Chat"):
        loaded_data = load_chat_from_s3(chosen_chat)
        if loaded_data is None:
            st.warning("No chat found or error loading.")
        else:
            st.session_state.messages = loaded_data
            st.success(f"Loaded chat '{chosen_chat}' from S3.")
            st.rerun()
else:
    st.write("No saved chats found in S3, or error listing them.")

# 5) Save/Load by name
st.subheader("Save/Load By Name")
chat_name = st.text_input("Chat Name", value="Default")
cols = st.columns(2)
with cols[0]:
    if st.button("Save Chat"):
        save_chat_to_s3(chat_name, st.session_state.messages)
with cols[1]:
    if st.button("Load Chat"):
        loaded_data = load_chat_from_s3(chat_name)
        if loaded_data is None:
            st.warning("No chat found with that name.")
        else:
            st.session_state.messages = loaded_data
            st.success(f"Loaded chat '{chat_name}' from S3.")
            st.rerun()

# 6) Clear
if st.button("Clear All Messages"):
    st.session_state.messages = []
    st.rerun()

# 7) Token usage
tk_usage = total_token_usage(st.session_state.messages)
st.write(f"Approx. Token Usage: **{tk_usage}**")

# -------------------------------------------------------------------
# MAIN Chat Display
# -------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    style_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
    with st.chat_message(msg["role"]):
        if msg["role"]=="user" and st.session_state.editing_idx == i:
            # Editing mode
            edited_text = st.text_area("Edit your message", msg["content"], key=f"edit_{i}")
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("Save Edit", key=f"save_{i}"):
                    st.session_state.messages[i]["content"] = edited_text
                    st.session_state.editing_idx = None
                    st.rerun()
            with c2:
                if st.button("Cancel", key=f"cancel_{i}"):
                    st.session_state.editing_idx = None
                    st.rerun()
        else:
            # Normal display
            st.markdown(
                f"<div class='chat-bubble {style_class}'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
            st.caption(f"{msg.get('timestamp','')}")

            # If assistant & user wants to see thinking
            if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
                st.markdown(f"<div class='thinking-box'>{msg['thinking']}</div>", unsafe_allow_html=True)

            # For user messages, show Edit/Delete
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

# -------------------------------------------------------------------
# BOTTOM INPUT
# -------------------------------------------------------------------
prompt = st.chat_input("Your message to Claude...")
if prompt:
    # Add user message
    user_msg = create_message("user", prompt)
    st.session_state.messages.append(user_msg)

    # Get Claude response
    client = get_bedrock_client()
    if client:
        ans_text, think_text = invoke_claude(
            client,
            st.session_state.messages,
            st.session_state.force_thinking,
            temperature,
            max_tokens
        )
        st.session_state.messages.append(
            create_message("assistant", ans_text, think_text)
        )

    st.rerun()