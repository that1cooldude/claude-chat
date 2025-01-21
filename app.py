import streamlit as st
import boto3
import json
import re
from datetime import datetime, timedelta
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
    page_title="Claude Chat (Artisan's Release)",
    page_icon="🎨",
    layout="wide"
)

# -----------------------------------------------------------------------------
# PROMPT TEMPLATES
# -----------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "Default": "You are Claude. Provide chain-of-thought if forced.",
    "Technical": "You are Claude. Focus on technical accuracy and provide detailed explanations with code examples when relevant.",
    "Statistical": "You are Claude. Focus on statistical analysis and data interpretation. Explain your statistical thinking process.",
    "Research": "You are Claude. Approach topics academically, citing your knowledge sources and explaining your reasoning."
}

# -----------------------------------------------------------------------------
# REFRESH CONTROL
# -----------------------------------------------------------------------------
if "refresh_state" not in st.session_state:
    st.session_state.refresh_state = {
        "last_update": datetime.now(),
        "counter": 0,
        "auto_refresh": False
    }

def should_update():
    """Control refresh frequency"""
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.refresh_state["last_update"]).seconds
    if time_diff > 2:  # Minimum 2 seconds between refreshes
        st.session_state.refresh_state["last_update"] = current_time
        return True
    return False

# -----------------------------------------------------------------------------
# BASIC CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
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
.stButton > button {
    width: 100%;
}
.chat-message {
    transition: all 0.3s ease;
}
</style>

<script>
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        const sendButton = Array.from(document.querySelectorAll('button')).find(
            button => button.innerText === 'Send Message'
        );
        if (sendButton) sendButton.click();
    }
});
</script>
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

    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False

    if "user_input_text" not in st.session_state:
        st.session_state.user_input_text = ""

    if "new_messages_since_last_update" not in st.session_state:
        st.session_state.new_messages_since_last_update = False

init_session()

def get_current_chat_data():
    return st.session_state.chats[st.session_state.current_chat]

# -----------------------------------------------------------------------------
# AWS CLIENTS WITH ENHANCED ERROR HANDLING
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
# S3 OPERATIONS WITH RETRY AND ERROR HANDLING
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def safe_s3_operation(operation_func):
    """Wrapper for S3 operations with better error handling"""
    try:
        return operation_func()
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            st.info("No previous chat found. Starting new conversation.")
            return None
        elif error_code == 'ThrottlingException':
            st.warning("AWS request throttled. Retrying...")
            raise
        else:
            st.error(f"AWS Error: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def save_chat_to_s3(chat_name, chat_data):
    def _save():
        s3 = get_s3_client()
        if not s3:
            return
        key = f"conversations/{chat_name}.json"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(chat_data, indent=2),
            ContentType="application/json"
        )
        return True
    
    result = safe_s3_operation(_save)
    if result:
        st.success(f"Saved chat '{chat_name}' to S3")

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_chat_from_s3(chat_name):
    def _load():
        s3 = get_s3_client()
        if not s3:
            return None
        key = f"conversations/{chat_name}.json"
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    
    return safe_s3_operation(_load)

def list_chats_s3():
    def _list():
        s3 = get_s3_client()
        if not s3:
            return []
        objs = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="conversations/")
        if "Contents" not in objs:
            return []
        return sorted([
            k.replace("conversations/","").replace(".json","")
            for k in [item["Key"] for item in objs["Contents"]]
            if k.endswith(".json")
        ])
    
    return safe_s3_operation(_list) or []

# -----------------------------------------------------------------------------
# TOKEN CALCULATION WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)  # Cache for 1 minute
def approximate_tokens(text: str) -> int:
    return len(text.split())

def total_token_usage(chat_data: dict):
    """Cached token counting"""
    total = 0
    for m in chat_data["messages"]:
        total += approximate_tokens(m["content"])
    total += approximate_tokens(chat_data.get("system_prompt",""))
    return total

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
            "content": [
                {"type":"text","text": msg["content"]}
            ]
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

# -----------------------------------------------------------------------------
# ENHANCED MESSAGE HANDLING
# -----------------------------------------------------------------------------
def handle_message(user_msg_str, chat_data, temperature, max_tokens):
    """Enhanced message handling with status indicators"""
    if not user_msg_str.strip():
        st.warning("No message entered. Please type something first.")
        return False
        
    with st.status("Processing message...", expanded=True) as status:
        # Add user message
        chat_data["messages"].append(create_message("user", user_msg_str))
        
        # Call Claude with progress indication
        status.update(label="Waiting for Claude's response...")
        client = get_bedrock_client()
        if client:
            try:
                ans_text, ans_think = invoke_claude(client, chat_data, temperature, max_tokens)
                chat_data["messages"].append(
                    create_message("assistant", ans_text, ans_think)
                )
                status.update(label="Response received!", state="complete")
                return True
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
                status.update(label="Error occurred", state="error")
                return False
    return False

# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------
col_chat, col_settings = st.columns([2,1], gap="large")

with col_settings:
    st.title("Claude Chat (Artisan's Release)")

    # Refresh Control
    st.subheader("Refresh Control")
    refresh_col1, refresh_col2 = st.columns([3, 1])
    with refresh_col1:
        if st.button(
            "↻ Refresh Chat",
            disabled=not should_update(),
            help="Refresh chat (available every 2 seconds)"
        ):
            st.rerun()
    with refresh_col2:
        if not should_update():
            time_diff = (datetime.now() - st.session_state.refresh_state["last_update"]).seconds
            st.write(f"Wait {2-time_diff}s")
    
    st.session_state.refresh_state["auto_refresh"] = st.toggle(
        "Enable auto-refresh",
        value=st.session_state.refresh_state.get("auto_refresh", False),
        help="Toggle automatic chat updates"
    )

    # Switch conversation
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

    # Create new conversation
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

    # Model Controls
    st.subheader("Model Controls")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1000)

    # System Prompt Templates
    st.subheader("System Prompt")
    template_choice = st.selectbox(
        "Prompt Template",
        options=["Custom"] + list(PROMPT_TEMPLATES.keys()),
        help="Select a pre-defined system prompt or use custom"
    )
    
    if template_choice != "Custom":
        chat_data["system_prompt"] = PROMPT_TEMPLATES[template_choice]
    
    chat_data["system_prompt"] = st.text_area(
        "Claude instructions",
        value=chat_data["system_prompt"],
        height=100
    )

    # Chain-of-Thought
    st.subheader("Chain-of-Thought")
    chat_data["force_thinking"] = st.checkbox(
        "Force chain-of-thought in <thinking>...",
        value=chat_data.get("force_thinking",False)
    )
    st.session_state.show_thinking = st.checkbox(
        "Show chain-of-thought",
        value=st.session_state.show_thinking
    )

    # Usage
    st.subheader("Token Usage")
    current_tokens = total_token_usage(chat_data)
    st.metric(
        "Approximate Tokens",
        value=current_tokens,
        delta=f"{current_tokens - st.session_state.get('last_token_count', current_tokens)}",
        help="Approximate token count for current conversation"
    )
    st.session_state.last_token_count = current_tokens

    # Export Chat
    st.subheader("Export Chat")
    if st.button("📥 Export Current Chat"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        chat_text = "# Chat Export\n\n"
        chat_text += f"System Prompt: {chat_data['system_prompt']}\n\n"
        for msg in chat_data["messages"]:
            chat_text += f"## {msg['role'].title()} ({msg['timestamp']})\n{msg['content']}\n\n"
            if msg.get('thinking'):
                chat_text += f"### Thinking Process\n{msg['thinking']}\n\n"
        
        st.download_button(
            "💾 Download Chat Log",
            chat_text,
            f"chat_export_{timestamp}.md",
            "text/markdown"
        )

    # Save/Load S3
    st.subheader("Save/Load to S3")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Chat"):
            save_chat_to_s3(st.session_state.current_chat, chat_data)
    with c2:
        if st.button("Load Chat"):
            loaded_data = load_chat_from_s3(st.session_state.current_chat)
            if loaded_data:
                st.session_state.chats[st.session_state.current_chat] = loaded_data
                st.success("Loaded conversation from S3.")

    # List S3 Chats
    s3_chats = list_chats_s3()
    if s3_chats:
        pick_s3 = st.selectbox("Load existing chat from S3", ["--select--"] + s3_chats)
        if pick_s3 != "--select--":
            if st.button("Load Selected S3 Chat"):
                data2 = load_chat_from_s3(pick_s3)
                if data2:
                    st.session_state.chats[pick_s3] = data2
                    st.session_state.current_chat = pick_s3
                    st.success(f"Loaded '{pick_s3}' from S3.")

    if st.button("Clear Current Chat"):
        chat_data["messages"] = []

with col_chat:
    st.header(f"Conversation: {st.session_state.current_chat}")

    # Chat display
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for i, msg in enumerate(chat_data["messages"]):
        bubble_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='timestamp'>{msg.get('timestamp','')}</div>", unsafe_allow_html=True)

        if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
            with st.expander("Chain-of-Thought"):
                st.markdown(
                    f"<div class='thinking-expander'><span class='thinking-text'>{msg['thinking']}</span></div>",
                    unsafe_allow_html=True
                )
    st.markdown("</div>", unsafe_allow_html=True)

    # Enhanced Message Input
    st.subheader("Type Your Message (Ctrl+Enter to send)")
    input_col1, input_col2 = st.columns([4, 1])
    with input_col1:
        st.session_state.user_input_text = st.text_area(
            "Message",
            value=st.session_state.user_input_text,
            key="message_input",
            height=100
        )
    with input_col2:
        st.markdown("#### Quick Links")
        if st.session_state.user_input_text:
            search_url = f"https://www.perplexity.ai/search?q={urllib.parse.quote(st.session_state.user_input_text)}"
            st.markdown(f"[🔍 Search Topic]({search_url})")

    send_col1, send_col2 = st.columns([4, 1])
    with send_col1:
        if st.button("Send Message", use_container_width=True):
            handle_message(
                st.session_state.user_input_text,
                chat_data,
                temperature,
                max_tokens
            )

# Auto-refresh handling
if st.session_state.refresh_state["auto_refresh"] and should_update():
    time.sleep(2)  # Prevent too frequent updates
    st.rerun()
