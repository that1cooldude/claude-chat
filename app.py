import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# ----------------------------------------------------------------------
# BASIC CONFIG
# ----------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")

# Use your Anthropic Claude inference profile ARN:
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

st.set_page_config(
    page_title="Claude Chat (Docs + Tools)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# SOME CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>
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

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
def init_session():
    if "messages" not in st.session_state:
        # Entire conversation stored here
        st.session_state.messages = []
    if "docs" not in st.session_state:
        # Loaded docs from S3: { "DocName": "DocContent", ... }
        st.session_state.docs = {}
    if "selected_docs" not in st.session_state:
        # Which docs to include in context
        st.session_state.selected_docs = []
    if "force_thinking" not in st.session_state:
        st.session_state.force_thinking = False
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
    if "editing_idx" not in st.session_state:
        st.session_state.editing_idx = None

init_session()

# ----------------------------------------------------------------------
# AWS CLIENTS
# ----------------------------------------------------------------------
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
        st.error(f"Error creating Bedrock client: {e}")
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
        st.error(f"Error creating S3 client: {e}")
        return None

# ----------------------------------------------------------------------
# S3 CHAT PERSISTENCE
# ----------------------------------------------------------------------
def save_chat(chat_name, messages):
    s3 = get_s3_client()
    if not s3:
        return
    key = f"conversations/{chat_name}.json"
    data = {"messages": messages}
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType="application/json"
        )
        st.success(f"Saved chat '{chat_name}' to s3://{S3_BUCKET_NAME}/{key}")
    except ClientError as e:
        st.error(f"Save error: {e}")

def load_chat(chat_name):
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
        st.error(f"Load error: {e}")
        return None

def list_saved_chats():
    """List chat files under 'conversations/' in S3."""
    s3 = get_s3_client()
    if not s3:
        return []
    try:
        objs = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="conversations/")
        if "Contents" not in objs:
            return []
        chat_names = []
        for item in objs["Contents"]:
            key = item["Key"]
            if key.endswith(".json"):
                name = key.replace("conversations/", "").replace(".json","")
                chat_names.append(name)
        return sorted(chat_names)
    except ClientError as e:
        st.error(f"List chats error: {e}")
        return []

# ----------------------------------------------------------------------
# DOCUMENTS (S3) - Simple approach (no chunking/embedding)
# ----------------------------------------------------------------------
def load_doc_from_s3(doc_name, s3_key):
    """Load a text doc from s3://bucket/<s3_key> into st.session_state.docs[doc_name]."""
    s3 = get_s3_client()
    if not s3:
        return
    try:
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        text_data = resp["Body"].read().decode("utf-8")
        st.session_state.docs[doc_name] = text_data
        st.success(f"Loaded doc '{doc_name}' from s3://{S3_BUCKET_NAME}/{s3_key}")
    except ClientError as e:
        st.error(f"Doc load error: {e}")

# ----------------------------------------------------------------------
# CLAUDE INVOCATION
# ----------------------------------------------------------------------
def approximate_tokens(text):
    return len(text.split())

def total_token_usage(messages):
    total = 0
    for m in messages:
        total += approximate_tokens(m["content"])
    return total

def create_message(role, content, thinking=""):
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

def build_bedrock_messages(messages, selected_docs, force_thinking):
    """
    We only use roles "user" & "assistant" to avoid schema issues.
    - If force_thinking is True, we prepend a user message instructing chain-of-thought usage.
    - For each doc in selected_docs, we add a user message: "DOC <doc_name>: <content>"
      so Claude sees that doc as context.
    - Then we add all user/assistant messages.
    """
    bedrock_msgs = []

    # 1) Force chain-of-thought if toggled
    if force_thinking:
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "You MUST use <thinking>...</thinking> to show chain-of-thought reasoning."}
            ]
        })

    # 2) Include selected docs as user messages
    for doc_name in selected_docs:
        doc_text = st.session_state.docs.get(doc_name, "")
        if doc_text.strip():
            doc_content = f"DOC {doc_name}:\n{doc_text}"
            bedrock_msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": doc_content}
                ]
            })

    # 3) Actual conversation
    for msg in messages:
        bedrock_msgs.append({
            "role": msg["role"],  # user or assistant
            "content": [{"type": "text", "text": msg["content"]}]
        })
    return bedrock_msgs

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, messages, selected_docs, force_thinking, temperature, max_tokens):
    with st.spinner("Claude is thinking..."):
        body_msgs = build_bedrock_messages(messages, selected_docs, force_thinking)
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
                "messages": body_msgs
            })
        )
        result = json.loads(resp["body"].read())
        if "content" in result:
            combined_text = "\n".join(seg["text"] for seg in result["content"] if seg["type"] == "text")
            # extract <thinking> if present
            thinking = ""
            match = re.search(r"<thinking>(.*?)</thinking>", combined_text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                combined_text = re.sub(r"<thinking>.*?</thinking>", "", combined_text, flags=re.DOTALL).strip()
            return combined_text, thinking
        return "No response from Claude.", ""

# ----------------------------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------------------------
st.title("Claude Chat (Docs + Tools, Minimal UI)")

# Tools: forcing chain-of-thought
st.session_state.force_thinking = st.checkbox(
    "Force Chain-of-Thought",
    value=st.session_state.force_thinking
)

# Tools: show/hide chain-of-thought
st.session_state.show_thinking = st.checkbox(
    "Show Chain-of-Thought",
    value=st.session_state.show_thinking
)

# Model settings
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.slider("Max Tokens", 100, 4096, 1000)

# Document Manager
st.subheader("Documents Manager")
doc_name_input = st.text_input("Name for the doc (like 'Doc1')", value="")
doc_key_input = st.text_input("S3 Key for doc", value="")
if st.button("Load Doc from S3"):
    if doc_name_input.strip() and doc_key_input.strip():
        load_doc_from_s3(doc_name_input, doc_key_input)

# Choose which docs to attach to conversation
if st.session_state.docs:
    all_doc_names = sorted(st.session_state.docs.keys())
    st.session_state.selected_docs = st.multiselect(
        "Select docs to attach to conversation",
        options=all_doc_names,
        default=st.session_state.selected_docs
    )
else:
    st.write("No docs loaded yet.")

# Save/Load Chats
st.subheader("Saved Chats in S3")
all_chat_files = list_saved_chats()
if all_chat_files:
    chosen_chat = st.selectbox("Pick a saved chat", all_chat_files)
    if st.button("Load Selected Chat"):
        loaded = load_chat(chosen_chat)
        if loaded is None:
            st.warning("That chat does not exist.")
        else:
            st.session_state.messages = loaded
            st.success(f"Loaded chat '{chosen_chat}' from S3.")
            st.rerun()
else:
    st.write("(No saved chats found or error listing them)")

chat_name = st.text_input("Chat Name to Save/Load", value="Default Chat")
col_s, col_l = st.columns(2)
with col_s:
    if st.button("Save Chat"):
        save_chat(chat_name, st.session_state.messages)
with col_l:
    if st.button("Load Chat"):
        loaded_data = load_chat(chat_name)
        if loaded_data is None:
            st.warning("No chat found with that name.")
        else:
            st.session_state.messages = loaded_data
            st.success(f"Loaded '{chat_name}' from S3.")
            st.rerun()

if st.button("Clear All Messages"):
    st.session_state.messages = []
    st.rerun()

tk_usage = total_token_usage(st.session_state.messages)
st.write(f"Approx Token Usage: **{tk_usage}**")

# ----------------------------------------------------------------------
# MAIN CHAT UI
# ----------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    style = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and st.session_state.editing_idx == i:
            # editing user message
            edited_text = st.text_area("Edit your message", value=msg["content"], key=f"edit_{i}")
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
            # normal bubble display
            st.markdown(f"<div class='chat-bubble {style}'>{msg['content']}</div>", unsafe_allow_html=True)
            st.caption(msg.get("timestamp",""))

            if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
                st.markdown(f"<div class='thinking-box'>{msg['thinking']}</div>", unsafe_allow_html=True)

            # user messages can be edited/deleted
            if msg["role"]=="user":
                cc1, cc2, _ = st.columns([1,1,8])
                with cc1:
                    if st.button("✏️ Edit", key=f"editbtn_{i}"):
                        st.session_state.editing_idx = i
                        st.rerun()
                with cc2:
                    if st.button("🗑️ Del", key=f"delbtn_{i}"):
                        st.session_state.messages.pop(i)
                        st.rerun()

# ----------------------------------------------------------------------
# BOTTOM CHAT INPUT
# ----------------------------------------------------------------------
user_input = st.chat_input("Your message to Claude...")
if user_input:
    # add user message
    st.session_state.messages.append(create_message("user", user_input))
    # invoke Claude
    client = get_bedrock_client()
    if client:
        ans, think = invoke_claude(
            client=client,
            messages=st.session_state.messages,
            selected_docs=st.session_state.selected_docs,
            force_thinking=st.session_state.force_thinking,
            temperature=temperature,
            max_tokens=max_tokens
        )
        st.session_state.messages.append(create_message("assistant", ans, think))
    st.rerun()