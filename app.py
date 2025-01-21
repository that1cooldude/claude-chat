import streamlit as st
import boto3
import json
import re
from datetime import datetime
import urllib.parse
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
    page_title="Claude Chat",
    page_icon="💭",
    layout="wide"
)

# -----------------------------------------------------------------------------
# BASIC CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.chat-container { display: flex; flex-direction: column; gap: 1rem; margin-top: 1rem; }
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
.stButton > button { width: 100%; }
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
                "system_prompt": "You are Claude, a helpful AI assistant.",
                "force_thinking": False
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
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
# S3 OPERATIONS
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
        st.success(f"Saved chat '{chat_name}' to S3")
    except Exception as e:
        st.error(f"Save error: {e}")

def load_chat_from_s3(chat_name):
    s3 = get_s3_client()
    if not s3:
        return None
    key = f"conversations/{chat_name}.json"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
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
        return sorted([
            k.replace("conversations/","").replace(".json","")
            for k in [item["Key"] for item in objs["Contents"]]
            if k.endswith(".json")
        ])
    except Exception as e:
        st.error(f"List error: {e}")
        return []

# -----------------------------------------------------------------------------
# CLAUDE INTERACTION
# -----------------------------------------------------------------------------
def build_bedrock_messages(chat_data):
    bedrock_msgs = []

    if chat_data.get("force_thinking", False):
        bedrock_msgs.append({
            "role":"user",
            "content":[{"type":"text","text":"Please include chain-of-thought in <thinking>...</thinking> if possible."}]
        })

    sys_prompt = chat_data.get("system_prompt","").strip()
    if sys_prompt:
        bedrock_msgs.append({
            "role":"user",
            "content":[{"type":"text","text":f"(System Prompt)\n{sys_prompt}"}]
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

def handle_message():
    """Process user message and get Claude's response"""
    user_msg = st.session_state.user_input_text.strip()
    chat_data = get_current_chat_data()
    
    if not user_msg:
        st.warning("Please enter a message.")
        return
    
    # Add user message
    chat_data["messages"].append(create_message("user", user_msg))
    
    # Clear input immediately
    st.session_state.user_input_text = ""
    
    # Get Claude's response
    with st.status("Getting response from Claude...", expanded=True) as status:
        client = get_bedrock_client()
        if client:
            try:
                response, thinking = invoke_claude(
                    client, 
                    chat_data,
                    st.session_state.get("temperature", 0.7),
                    st.session_state.get("max_tokens", 1000)
                )
                # Add assistant message
                chat_data["messages"].append(
                    create_message("assistant", response, thinking)
                )
                status.update(label="Response received!", state="complete")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                status.update(label="Error occurred", state="error")

# -----------------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------------
col_chat, col_settings = st.columns([2,1], gap="large")

with col_settings:
    st.title("Claude Chat")

    # Model Controls
    st.subheader("Settings")
    st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.session_state["max_tokens"] = st.slider("Max Tokens", 100, 4096, 1000)
    
    # System Prompt
    st.subheader("System Prompt")
    chat_data = get_current_chat_data()
    chat_data["system_prompt"] = st.text_area(
        "Instructions for Claude",
        value=chat_data.get("system_prompt",""),
        height=100
    )

    # Conversations
    st.subheader("Conversations")
    chat_keys = list(st.session_state.chats.keys())
    current_idx = chat_keys.index(st.session_state.current_chat)
    chosen_chat = st.selectbox("Select Chat", chat_keys, index=current_idx)
    if chosen_chat != st.session_state.current_chat:
        st.session_state.current_chat = chosen_chat

    col1, col2 = st.columns(2)
    with col1:
        new_chat = st.text_input("New Chat Name")
        if st.button("Create") and new_chat:
            if new_chat not in st.session_state.chats:
                st.session_state.chats[new_chat] = {
                    "messages": [],
                    "system_prompt": chat_data["system_prompt"],
                    "force_thinking": False
                }
                st.session_state.current_chat = new_chat
                st.experimental_rerun()

    # S3 Operations
    st.subheader("Save/Load")
    s3col1, s3col2 = st.columns(2)
    with s3col1:
        if st.button("Save to S3"):
            save_chat_to_s3(st.session_state.current_chat, chat_data)
    with s3col2:
        if st.button("Load from S3"):
            loaded = load_chat_from_s3(st.session_state.current_chat)
            if loaded:
                st.session_state.chats[st.session_state.current_chat] = loaded
                st.success("Loaded chat")
                st.experimental_rerun()

    # Thinking Display
    st.subheader("Chain of Thought")
    chat_data["force_thinking"] = st.checkbox(
        "Request thinking process",
        value=chat_data.get("force_thinking", False)
    )
    st.session_state.show_thinking = st.checkbox(
        "Show thinking",
        value=st.session_state.show_thinking
    )

    if st.button("Clear Chat"):
        if st.session_state.current_chat in st.session_state.chats:
            st.session_state.chats[st.session_state.current_chat]["messages"] = []
            st.experimental_rerun()

with col_chat:
    st.header(f"Chat: {st.session_state.current_chat}")
    
    # Display Messages
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    for msg in chat_data["messages"]:
        # Message bubble
        bubble_class = "assistant-bubble" if msg["role"] == "assistant" else "user-bubble"
        st.markdown(
            f"<div class='{bubble_class}'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
        
        # Timestamp
        st.markdown(
            f"<div class='timestamp'>{msg.get('timestamp', '')}</div>",
            unsafe_allow_html=True
        )
        
        # Thinking process for assistant messages
        if (msg["role"] == "assistant" and 
            st.session_state.show_thinking and 
            msg.get("thinking")):
            with st.expander("Chain-of-Thought"):
                st.markdown(
                    f"<div class='thinking-expander'>"
                    f"<span class='thinking-text'>{msg['thinking']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Message Input
    st.text_area(
        "Message",
        key="user_input_text",
        height=100,
        placeholder="Type your message here... (Ctrl+Enter to send)"
    )
    
    # Quick Links
    if st.session_state.user_input_text:
        search_url = (
            "https://www.perplexity.ai/search?q=" +
            urllib.parse.quote(st.session_state.user_input_text)
        )
        st.markdown(f"[🔍 Search this topic]({search_url})")
    
    # Send Button
    if st.button("Send Message", use_container_width=True):
        handle_message()
