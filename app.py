import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Claude Chat (Improved UI)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# STYLING (with feedback from a hypothetical UI designer)
# --------------------------------------------------------------------
st.markdown("""
<style>
/* Larger container for the messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
}

/* Bubbles */
.chat-bubble {
    padding: 12px 16px;
    border-radius: 10px;
    margin-bottom: 8px;
    line-height: 1.4;
    max-width: 80%;
    word-wrap: break-word;
}
.user-bubble {
    background-color: #2e3136;
    color: #fff;
    align-self: flex-end; /* Float to the right */
}
.assistant-bubble {
    background-color: #36393f;
    color: #fff;
    align-self: flex-start; /* Float to the left */
}

/* Thinking Expander style */
.thinking-expander {
    margin-top: 4px;
    background-color: #1e1e2e;
    border-left: 3px solid #ffd700;
    border-radius: 5px;
    padding: 6px 10px;
    font-style: italic;
}

/* Timestamp style */
.timestamp {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-top: 4px;
    text-align: right;
}

/* For the "Claude is typing..." indicator */
.typing-indicator {
    margin-top: 8px;
    font-weight: bold;
    color: #ffd700;
    animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------
def init_session():
    if "chats" not in st.session_state:
        # Example: st.session_state.chats = { "Default": { "messages":[], "system_prompt":"", "force_thinking":False } }
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
    if "typing" not in st.session_state:
        st.session_state.typing = False  # indicates "Claude is typing..."
    if "prompt_cache" not in st.session_state:
        # store a list of recent prompts
        st.session_state.prompt_cache = []

init_session()

# Convenience
def current_chat_data():
    return st.session_state.chats[st.session_state.current_chat]

init_session()

# --------------------------------------------------------------------
# AWS CLIENTS
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# S3 PERSISTENCE
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# TOKEN COUNT
# --------------------------------------------------------------------
def approximate_tokens(text: str) -> int:
    return len(text.split())

def total_token_usage(chat_data: dict):
    # sum over all messages
    total = 0
    for m in chat_data["messages"]:
        total += approximate_tokens(m["content"])
    # plus system prompt
    total += approximate_tokens(chat_data.get("system_prompt",""))
    return total

# --------------------------------------------------------------------
# BUILDING MESSAGES
# --------------------------------------------------------------------
def build_bedrock_messages(chat_data):
    """
    We only use 'user' / 'assistant'.
    If force_thinking is True, prepend a user message telling Claude to produce <thinking>.
    Then system_prompt is also a user message.
    Then the actual conversation.
    """
    bedrock_msgs = []

    if chat_data.get("force_thinking", False):
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type":"text", "text":"Please include chain-of-thought in <thinking>...</thinking> if possible."}
            ]
        })

    system_prompt = chat_data.get("system_prompt","").strip()
    if system_prompt:
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type":"text","text": f"(System Prompt)\n{system_prompt}"}
            ]
        })

    for msg in chat_data["messages"]:
        bedrock_msgs.append({
            "role": msg["role"],
            "content": [
                {"type":"text","text": msg["content"]}
            ]
        })

    return bedrock_msgs

# --------------------------------------------------------------------
# CALLING CLAUDE
# --------------------------------------------------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, chat_data, temperature, max_tokens):
    st.session_state.typing = True  # show "Claude is typing..."
    try:
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
            # parse <thinking>
            think = ""
            match = re.search(r"<thinking>(.*?)</thinking>", combined, re.DOTALL)
            if match:
                think = match.group(1).strip()
                combined = re.sub(r"<thinking>.*?</thinking>", "", combined, flags=re.DOTALL).strip()
            return combined, think
        return "No response from Claude.", ""
    finally:
        st.session_state.typing = False  # done typing

def create_message(role, content, thinking=""):
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# --------------------------------------------------------------------
# UI LAYOUT
# --------------------------------------------------------------------
col_chat, col_settings = st.columns([2,1], gap="large")

# =======================
# RIGHT COLUMN: Settings
# =======================
with col_settings:
    st.header("Claude Chat Settings")

    # Select conversation
    chat_keys = list(st.session_state.chats.keys())
    chosen_chat = st.selectbox(
        "Choose Conversation",
        options=chat_keys,
        index=chat_keys.index(st.session_state.current_chat)
        if st.session_state.current_chat in chat_keys
        else 0
    )
    if chosen_chat != st.session_state.current_chat:
        st.session_state.current_chat = chosen_chat
        st.rerun()

    # Create conversation
    new_chat_name = st.text_input("New Chat Name", "")
    if st.button("Create New Chat"):
        if new_chat_name and new_chat_name not in st.session_state.chats:
            st.session_state.chats[new_chat_name] = {
                "messages": [],
                "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                "force_thinking": False
            }
            st.session_state.current_chat = new_chat_name
            st.rerun()

    # Model Controls
    st.subheader("Model Controls")
    temp = st.slider("Temperature", 0.0, 1.0, 0.7)
    maxtok = st.slider("Max Tokens", 100, 4096, 1000)

    # System Prompt
    st.subheader("System Prompt")
    chat_data = current_chat_data()
    chat_data["system_prompt"] = st.text_area(
        "Claude's instruction text",
        value=chat_data.get("system_prompt","")
    )

    # Chain-of-Thought
    st.subheader("Chain-of-Thought")
    chat_data["force_thinking"] = st.checkbox(
        "Force chain-of-thought in <thinking>...",
        value=chat_data.get("force_thinking", False)
    )
    st.session_state.show_thinking = st.checkbox("Show chain-of-thought", value=st.session_state.show_thinking)

    # Usage
    st.subheader("Usage")
    tcount = total_token_usage(chat_data)
    st.write(f"Approx. tokens: **{tcount}**")

    # Prompt Caching
    st.subheader("Prompt Caching")
    if len(st.session_state.prompt_cache) > 0:
        selected_prompt = st.selectbox("Cached Prompts", ["--select--"] + st.session_state.prompt_cache)
        if selected_prompt != "--select--":
            if st.button("Use Cached Prompt"):
                # Insert that prompt into the chat box or directly as a user message
                pass  # We'll handle usage below in the left column.

    # Save/Load S3
    st.subheader("Save / Load to S3")
    rowS, rowL = st.columns(2)
    with rowS:
        if st.button("Save Chat to S3"):
            save_chat_to_s3(st.session_state.current_chat, chat_data)
    with rowL:
        if st.button("Load Chat from S3"):
            loaded = load_chat_from_s3(st.session_state.current_chat)
            if loaded is None:
                st.warning(f"No record in S3 for '{st.session_state.current_chat}'")
            else:
                st.session_state.chats[st.session_state.current_chat] = loaded
                st.success("Loaded from S3.")
                st.rerun()

    # List S3 Chats
    s3_chats = list_chats_s3()
    if s3_chats:
        sel_s3 = st.selectbox("Load existing S3 chat", ["--select--"] + s3_chats)
        if sel_s3 != "--select--":
            if st.button("Load chosen S3 conversation"):
                data2 = load_chat_from_s3(sel_s3)
                if data2:
                    st.session_state.chats[sel_s3] = data2
                    st.session_state.current_chat = sel_s3
                    st.success(f"Loaded '{sel_s3}' from S3.")
                    st.rerun()

    # Clear
    if st.button("Clear Current Chat"):
        chat_data["messages"] = []
        st.rerun()

# ======================
# LEFT COLUMN: Chat UI
# ======================
with col_chat:
    st.header(f"Claude Chat ({st.session_state.current_chat})")

    # Show conversation
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for i, msg in enumerate(chat_data["messages"]):
        bubble_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
        # If user is editing a message
        if msg["role"]=="user" and st.session_state.editing_idx == i:
            new_msg = st.text_area("Edit message", value=msg["content"], key=f"edit_{i}")
            colA, colB = st.columns([1,1])
            with colA:
                if st.button("Save", key=f"save_{i}"):
                    msg["content"] = new_msg
                    st.session_state.editing_idx = None
                    st.rerun()
            with colB:
                if st.button("Cancel", key=f"cancel_{i}"):
                    st.session_state.editing_idx = None
                    st.rerun()
        else:
            # Normal bubble
            st.markdown(f"<div class='chat-bubble {bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            # Timestamp
            st.markdown(f"<div class='timestamp'>{msg.get('timestamp','')}</div>", unsafe_allow_html=True)

            # If assistant + show_thinking + thinking text
            if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
                # We put the chain-of-thought in an expander
                with st.expander("Chain-of-Thought"):
                    st.markdown(f"<div class='thinking-expander'>{msg['thinking']}</div>", unsafe_allow_html=True)

            # For user messages, show edit / delete
            if msg["role"]=="user":
                ecol1, ecol2, _ = st.columns([1,1,8])
                with ecol1:
                    if st.button("✏️ Edit", key=f"editbtn_{i}"):
                        st.session_state.editing_idx = i
                        st.rerun()
                with ecol2:
                    if st.button("🗑️ Del", key=f"delbtn_{i}"):
                        chat_data["messages"].pop(i)
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # If Claude is "typing..."
    if st.session_state.typing:
        st.markdown("<div class='typing-indicator'>Claude is typing...</div>", unsafe_allow_html=True)

    # Chat input
    user_prompt = st.text_input("Your message to Claude...")
    if user_prompt:
        # Add to prompt_cache
        st.session_state.prompt_cache.insert(0, user_prompt)
        # keep only 10 cached prompts
        st.session_state.prompt_cache = st.session_state.prompt_cache[:10]

        # Add user message
        chat_data["messages"].append(create_message("user", user_prompt))

        # Call Claude
        client = get_bedrock_client()
        if client:
            answer, thinking_str = invoke_claude(
                client=client,
                chat_data=chat_data,
                temperature=temp,
                max_tokens=maxtok
            )
            chat_data["messages"].append(create_message("assistant", answer, thinking_str))

        st.rerun()
