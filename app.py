import streamlit as st
import boto3
import json
import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from botocore.exceptions import ClientError

# -------------------------------
# CONFIG - Fill or use st.secrets
# -------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stChat message {
        background-color: #2e3136;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    .message-timestamp {
        font-size: 0.8em;
        color: rgba(255,255,255,0.5);
        margin-top: 5px;
    }
    @media (max-width: 768px) {
        .stButton button {
            padding: 0.5rem !important;
            width: auto !important;
        }
        .row-widget.stButton {
            margin: 0 !important;
        }
    }
    .thinking-container {
        background-color: #1e1e2e; 
        border-left: 3px solid #ffd700; 
        padding: 10px; 
        margin-top: 10px; 
        font-style: italic;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Default": {
                "messages": [],
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "system_prompt": "Please include chain-of-thought in <thinking>...</thinking>."
                },
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
    if "knowledge" not in st.session_state:
        st.session_state.knowledge = ""

    if st.session_state.current_chat not in st.session_state.chats:
        st.session_state.current_chat = "Default"
        if "Default" not in st.session_state.chats:
            st.session_state.chats["Default"] = {
                "messages": [],
                "settings": {
                    "temperature": 0.7,
                    "max_tokens": 1000,
                    "system_prompt": "Please include chain-of-thought in <thinking>...</thinking>."
                },
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

init_session_state()

@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {e}")
        return None

@st.cache_resource
def get_s3_client():
    try:
        return boto3.client(
            service_name='s3',
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize S3 client: {e}")
        return None

def save_chat_to_s3(chat_name, chat_data, bucket_name):
    s3_client = get_s3_client()
    if not s3_client:
        return
    key = f"conversations/{chat_name}.json"
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(chat_data, indent=2),
            ContentType="application/json"
        )
        st.success(f"Saved '{chat_name}' to s3://{bucket_name}/{key}")
    except ClientError as e:
        st.error(f"Error saving chat: {e}")

def load_chat_from_s3(chat_name, bucket_name):
    s3_client = get_s3_client()
    if not s3_client:
        return None
    key = f"conversations/{chat_name}.json"
    try:
        resp = s3_client.get_object(Bucket=bucket_name, Key=key)
        data = resp['Body'].read().decode('utf-8')
        return json.loads(data)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        st.error(f"Error loading chat: {e}")
        return None

def load_kb_from_s3(kb_key, bucket_name):
    s3_client = get_s3_client()
    if not s3_client:
        return ""
    try:
        resp = s3_client.get_object(Bucket=bucket_name, Key=kb_key)
        return resp['Body'].read().decode('utf-8')
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            st.warning(f"No KB at s3://{bucket_name}/{kb_key}")
            return ""
        st.error(f"KB load error: {e}")
        return ""

def build_bedrock_messages(conversation, settings, knowledge):
    msgs = []
    sys_prompt = settings.get("system_prompt", "")
    msgs.append({
        "role": "system",
        "content": [{"type": "text", "text": sys_prompt}]
    })
    if knowledge.strip():
        kb_txt = f"Additional knowledge:\n{knowledge}"
        msgs.append({
            "role": "system",
            "content": [{"type": "text", "text": kb_txt}]
        })
    for msg in conversation:
        msgs.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })
    return msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
def get_chat_response(client, conversation, settings, knowledge):
    try:
        with st.spinner("Claude is thinking..."):
            payload = build_bedrock_messages(conversation, settings, knowledge)
            resp = client.invoke_model(
                modelId=MODEL_ARN,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "top_k": 250,
                    "top_p": 0.999,
                    "messages": payload
                })
            )
            body = json.loads(resp['body'].read())
            if 'content' in body:
                parts = []
                for p in body['content']:
                    if p.get('type') == 'text':
                        parts.append(p['text'])
                text = "\n".join(parts).strip()
                thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
                if thinking_match:
                    extracted = thinking_match.group(1).strip()
                    visible = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
                else:
                    extracted, visible = "", text
                return visible, extracted
            else:
                return "No response content.", ""
    except Exception as e:
        st.error(f"Error calling Claude: {e}")
        return None, None

def process_message(content, role, thinking=""):
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }

def approximate_tokens(text):
    return len(text.split())

def total_token_usage(convo):
    return sum(approximate_tokens(m["content"]) for m in convo)

with st.sidebar:
    st.title("Chat Settings")
    try:
        st.subheader("Conversations")
        new_chat = st.text_input("New Chat Name", key="new_chat_input")
        if st.button("Create Chat"):
            if new_chat and new_chat not in st.session_state.chats:
                st.session_state.chats[new_chat] = {
                    "messages": [],
                    "settings": {
                        "temperature": 0.7,
                        "max_tokens": 1000,
                        "system_prompt": "Please include chain-of-thought in <thinking>...</thinking>."
                    },
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat
                st.experimental_rerun()

        st.session_state.current_chat = st.selectbox(
            "Select Chat",
            options=list(st.session_state.chats.keys()),
            index=list(st.session_state.chats.keys()).index(st.session_state.current_chat)
            if st.session_state.current_chat in st.session_state.chats
            else 0
        )
        current_chat = st.session_state.chats[st.session_state.current_chat]

        st.subheader("Model Settings")
        current_chat["settings"]["temperature"] = st.slider(
            "Temperature", 0.0, 1.0,
            current_chat["settings"].get("temperature", 0.7)
        )
        current_chat["settings"]["max_tokens"] = st.slider(
            "Max Tokens", 100, 4096,
            current_chat["settings"].get("max_tokens", 1000)
        )
        st.subheader("System Prompt")
        current_chat["settings"]["system_prompt"] = st.text_area(
            "Claude sees this first.",
            value=current_chat["settings"].get("system_prompt", ""),
        )

        st.subheader("Thinking Process")
        st.session_state.show_thinking = st.checkbox("Show Thinking", value=st.session_state.show_thinking)

        st.subheader("Approx Token Usage")
        tk_count = total_token_usage(current_chat["messages"])
        st.write(f"Tokens: **{tk_count}**")

        st.subheader("Perplexity Search")
        p_query = st.text_input("Search Query")
        if st.button("Search Perplexity"):
            if p_query.strip():
                link = p_query.strip().replace(" ", "+")
                url = f"https://www.perplexity.ai/search?q={link}"
                st.markdown(f"[Open Perplexity]({url})", unsafe_allow_html=True)
            else:
                st.warning("Enter a query first.")

        st.subheader("Knowledge Base")
        kb_key = st.text_input("S3 Key for KB (e.g. 'mykb/finance.txt')")
        if st.button("Load KB"):
            kb_text = load_kb_from_s3(kb_key, S3_BUCKET_NAME)
            if kb_text:
                st.session_state.knowledge = kb_text
                st.success("KB loaded. It's appended to Claude's context.")
            else:
                st.session_state.knowledge = ""

        st.subheader("Cloud Storage")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save Chat"):
                save_chat_to_s3(st.session_state.current_chat, current_chat, S3_BUCKET_NAME)
        with c2:
            if st.button("Load Chat"):
                loaded = load_chat_from_s3(st.session_state.current_chat, S3_BUCKET_NAME)
                if loaded is not None:
                    st.session_state.chats[st.session_state.current_chat] = loaded
                    st.success(f"Loaded {st.session_state.current_chat} from S3.")
                    st.experimental_rerun()
                else:
                    st.warning("No chat found in S3 for that name.")

        st.subheader("Clear Conversation")
        if st.button("Clear Current Chat"):
            if len(current_chat["messages"]) > 0:
                if st.button("Confirm Clear?"):
                    current_chat["messages"] = []
                    st.experimental_rerun()
            else:
                current_chat["messages"] = []
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Error in sidebar: {e}")
        init_session_state()

st.title(f"Claude Chat - {st.session_state.current_chat}")

try:
    for idx, msg in enumerate(current_chat["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user" and st.session_state.editing_message == idx:
                edited = st.text_area("Edit your message", msg["content"], key=f"edit_{idx}")
                cA, cB = st.columns([1,1])
                with cA:
                    if st.button("Save", key=f"save_{idx}"):
                        current_chat["messages"][idx]["content"] = edited
                        st.session_state.editing_message = None
                        st.experimental_rerun()
                with cB:
                    if st.button("Cancel", key=f"cancel_{idx}"):
                        st.session_state.editing_message = None
                        st.experimental_rerun()
            else:
                st.markdown(msg["content"])
                st.caption(f"Time: {msg['timestamp']}")
                if msg["role"] == "assistant" and msg.get("thinking") and st.session_state.show_thinking:
                    with st.expander("Thinking Process"):
                        st.markdown(
                            f"<div class='thinking-container'>{msg['thinking']}</div>",
                            unsafe_allow_html=True
                        )
                if msg["role"] == "user":
                    col1, col2, _ = st.columns([1,1,8])
                    with col1:
                        if st.button("✏️", key=f"editbtn_{idx}"):
                            st.session_state.editing_message = idx
                            st.experimental_rerun()
                    with col2:
                        if st.button("🗑️", key=f"delbtn_{idx}"):
                            current_chat["messages"].pop(idx)
                            st.experimental_rerun()

    prompt = st.chat_input("Message Claude...")
    if prompt:
        current_chat["messages"].append(process_message(prompt, "user"))
        bd_client = get_bedrock_client()
        if bd_client:
            ans, think = get_chat_response(
                bd_client, current_chat["messages"],
                current_chat["settings"], st.session_state.knowledge
            )
            if ans is not None:
                current_chat["messages"].append(process_message(ans, "assistant", think))
        st.experimental_rerun()

except Exception as e:
    st.error(f"Error: {e}")
    if st.button("Reset App"):
        init_session_state()
        st.experimental_rerun()
