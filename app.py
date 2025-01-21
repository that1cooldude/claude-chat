import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Claude Chat (Derek Breese)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------
# BASIC STYLING
# -----------------------------------------------------------------
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
}
.chat-bubble {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    line-height: 1.4;
}
.user-bubble {
    background-color: #2e3136;
    align-self: flex-end;
}
.assistant-bubble {
    background-color: #36393f;
    align-self: flex-start;
}
.thinking-box {
    background-color: #1e1e2e; 
    border-left: 3px solid #ffd700; 
    padding: 10px; 
    margin-top: 10px; 
    font-style: italic;
    border-radius: 5px;
}
.timestamp {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.6);
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------
def init_session_state():
    # st.session_state.chats = { "chatName": { "messages": [], "system_prompt": "...", "force_thinking": bool } }
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
    if "editing_idx" not in st.session_state:
        st.session_state.editing_idx = None

init_session_state()

# -----------------------------------------------------------------
# AWS CLIENTS
# -----------------------------------------------------------------
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
        st.error(f"Bedrock client init error: {e}")
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
        st.error(f"S3 client init error: {e}")
        return None

# -----------------------------------------------------------------
# S3 PERSISTENCE
# -----------------------------------------------------------------
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
        names = []
        for item in objs["Contents"]:
            k = item["Key"]
            if k.endswith(".json"):
                base = k.replace("conversations/","").replace(".json","")
                names.append(base)
        return sorted(names)
    except ClientError as e:
        st.error(f"List error: {e}")
        return []

# -----------------------------------------------------------------
# TOKEN COUNT
# -----------------------------------------------------------------
def approximate_tokens(text):
    return len(text.split())

def total_token_usage(chat_data):
    # sum over all messages
    total = sum(approximate_tokens(m["content"]) for m in chat_data["messages"])
    # plus system prompt
    total += approximate_tokens(chat_data.get("system_prompt",""))
    return total

# -----------------------------------------------------------------
# BUILDING MESSAGES
# -----------------------------------------------------------------
def build_bedrock_messages(chat_data):
    """
    We'll only use 'user' / 'assistant' roles.
    If force_thinking is True, prepend a user message telling Claude to produce <thinking>.
    Then insert system_prompt as if user said it.
    Then add the actual conversation.
    """
    bedrock_msgs = []

    if chat_data.get("force_thinking", False):
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type":"text", "text":"Please include chain-of-thought in <thinking>...</thinking> if possible."}
            ]
        })

    sprompt = chat_data.get("system_prompt","").strip()
    if sprompt:
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type":"text", "text": f"(System Prompt)\n{sprompt}"}
            ]
        })

    for msg in chat_data["messages"]:
        bedrock_msgs.append({
            "role": msg["role"], 
            "content": [{"type":"text","text": msg["content"]}]
        })
    return bedrock_msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, chat_data, temperature, max_tokens):
    bedrock_msgs = build_bedrock_messages(chat_data)
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
            if seg.get("type")=="text":
                text_bits.append(seg["text"])
        combined = "\n".join(text_bits).strip()
        # extract <thinking>
        think = ""
        match = re.search(r"<thinking>(.*?)</thinking>", combined, re.DOTALL)
        if match:
            think = match.group(1).strip()
            combined = re.sub(r"<thinking>.*?</thinking>", "", combined, flags=re.DOTALL).strip()
        return combined, think
    return ("No response from Claude.", "")

def create_message(role, content, thinking=""):
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# -----------------------------------------------------------------
# LAYOUT (TWO COLUMNS)
# -----------------------------------------------------------------
col_chat, col_settings = st.columns([2,1], gap="large")

# 1) RIGHT COLUMN: SETTINGS
with col_settings:
    st.header("Settings / Actions")

    # Chat selection
    chat_keys = list(st.session_state.chats.keys())
    chosen_chat = st.selectbox(
        "Select Conversation",
        options=chat_keys,
        index=chat_keys.index(st.session_state.current_chat)
        if st.session_state.current_chat in chat_keys
        else 0
    )
    if chosen_chat != st.session_state.current_chat:
        st.session_state.current_chat = chosen_chat
        st.rerun()

    # Create new conversation
    new_chat_name = st.text_input("New Chat Name", "")
    if st.button("Create Conversation"):
        if new_chat_name and new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                "force_thinking": False
            }
            st.session_state.current_chat = new_chat_name
            st.rerun()

    current_chat = st.session_state.chats[st.session_state.current_chat]

    st.subheader("Model Settings")
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    maxtok = st.slider("Max Tokens", 100, 4096, 1000)

    st.subheader("System Prompt")
    current_chat["system_prompt"] = st.text_area(
        "Claude's instructions",
        value=current_chat.get("system_prompt","")
    )

    st.subheader("Chain-of-Thought")
    current_chat["force_thinking"] = st.checkbox(
        "Force chain-of-thought in <thinking>...",
        value=current_chat.get("force_thinking", False)
    )
    st.session_state.show_thinking = st.checkbox(
        "Show chain-of-thought",
        value=st.session_state.show_thinking
    )

    # Usage
    st.subheader("Usage")
    tok_count = total_token_usage(current_chat)
    st.write(f"Approx. tokens: **{tok_count}**")

    # Save / Load
    st.subheader("Save / Load to S3")
    rowA, rowB = st.columns(2)
    with rowA:
        if st.button("Save to S3"):
            save_chat_to_s3(st.session_state.current_chat, current_chat)
    with rowB:
        if st.button("Load from S3"):
            loaded_data = load_chat_from_s3(st.session_state.current_chat)
            if loaded_data is None:
                st.warning("No S3 record found.")
            else:
                st.session_state.chats[st.session_state.current_chat] = loaded_data
                st.success("Loaded from S3.")
                st.rerun()

    # List S3 Chats
    available_s3_chats = list_chats_s3()
    if available_s3_chats:
        selected_s3_chat = st.selectbox("Load existing S3 chat", ["--select--"] + available_s3_chats)
        if selected_s3_chat != "--select--":
            if st.button("Load chosen S3 Chat"):
                data2 = load_chat_from_s3(selected_s3_chat)
                if data2:
                    st.session_state.chats[selected_s3_chat] = data2
                    st.session_state.current_chat = selected_s3_chat
                    st.rerun()

    # Clear conversation
    if st.button("Clear Current Chat"):
        current_chat["messages"] = []
        st.rerun()

# 2) LEFT COLUMN: THE CHAT ITSELF
with col_chat:
    st.header(f"Claude Chat ({st.session_state.current_chat})")

    # Display messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for i, msg in enumerate(current_chat["messages"]):
        bubble_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
        
        # EDITING logic for user messages
        if msg["role"]=="user" and st.session_state.editing_idx == i:
            new_text = st.text_area("Edit message", value=msg["content"], key=f"edit_{i}")
            c_save, c_cancel = st.columns([1,1])
            with c_save:
                if st.button("Save", key=f"save_{i}"):
                    current_chat["messages"][i]["content"] = new_text
                    st.session_state.editing_idx = None
                    st.rerun()
            with c_cancel:
                if st.button("Cancel", key=f"cancel_{i}"):
                    st.session_state.editing_idx = None
                    st.rerun()
        else:
            # Normal display
            st.markdown(f"<div class='chat-bubble {bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='timestamp'>{msg['timestamp']}</div>", unsafe_allow_html=True)

            # If assistant and we want to show chain-of-thought
            if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
                st.markdown(f"<div class='thinking-box'>{msg['thinking']}</div>", unsafe_allow_html=True)

            # For user messages, show Edit / Delete
            if msg["role"]=="user":
                col_e, col_d, _ = st.columns([1,1,8])
                with col_e:
                    if st.button("✏️ Edit", key=f"editbtn_{i}"):
                        st.session_state.editing_idx = i
                        st.rerun()
                with col_d:
                    if st.button("🗑️ Del", key=f"delbtn_{i}"):
                        current_chat["messages"].pop(i)
                        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input("Your message to Claude...")
    if user_input:
        # Add user message
        current_chat["messages"].append(create_message("user", user_input))
        # Invoke Claude
        client = get_bedrock_client()
        if client:
            ans_text, think_text = invoke_claude(client, current_chat, temperature=temp, max_tokens=maxtok)
            current_chat["messages"].append(create_message("assistant", ans_text, think_text))
        st.rerun()
