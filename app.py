import streamlit as st
import boto3
import json
import re
import base64
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# --------------------------------------------------------------------
# CONFIG - read from st.secrets or fill in directly
# --------------------------------------------------------------------
AWS_REGION = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
S3_BUCKET_NAME = st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket")
MODEL_ARN = (
    "arn:aws:bedrock:us-east-2:127214158930:"
    "inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
)

# A rough estimate of your monthly or personal AWS credits in USD
# You can store something like `AWS_CREDITS = "100"` in secrets and parse float.
DEFAULT_AWS_CREDITS = float(st.secrets.get("AWS_CREDITS", "200.0"))

st.set_page_config(
    page_title="Claude Chat (Single-File, Enhanced)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------
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
.image-bubble {
    background-color: #1e1e2e;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px;
    margin: 15px 0;
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

# --------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------
def init_session():
    # We store multiple conversations in st.session_state["chats"] as {chat_name -> {messages:[], system_prompt:...}}
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Default": {
                "messages": [],
                "system_prompt": "You are Claude. Provide detailed reasoning in <thinking>...</thinking> if forced.",
                "force_thinking": False
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = False
    if "editing_idx" not in st.session_state:
        st.session_state.editing_idx = None
    if "aws_credits" not in st.session_state:
        # The user can set how many credits they have or want to track
        st.session_state.aws_credits = DEFAULT_AWS_CREDITS

init_session()

# --------------------------------------------------------------------
# AWS Clients
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

# --------------------------------------------------------------------
# S3 Chat Persistence
# --------------------------------------------------------------------
def save_chat_to_s3(chat_name, chat_data):
    s3 = get_s3_client()
    if not s3:
        return
    # chat_data is { "messages": [...], "system_prompt": "...", "force_thinking": bool }
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
        raw_data = resp["Body"].read().decode("utf-8")
        data = json.loads(raw_data)
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
        results = []
        for item in objs["Contents"]:
            key = item["Key"]
            if key.endswith(".json"):
                name = key.replace("conversations/", "").replace(".json","")
                results.append(name)
        return sorted(results)
    except ClientError as e:
        st.error(f"List chats error: {e}")
        return []

# --------------------------------------------------------------------
# HELPER: Approx tokens
# --------------------------------------------------------------------
def approximate_tokens(text: str) -> int:
    return len(text.split())

def total_token_usage(chat_data):
    # sum across all messages' content
    total = 0
    for m in chat_data["messages"]:
        total += approximate_tokens(m["content"])
    # also add system prompt if it exists
    total += approximate_tokens(chat_data.get("system_prompt",""))
    return total

# --------------------------------------------------------------------
# BUILD MESSAGES FOR CLAUDE
# --------------------------------------------------------------------
def build_bedrock_messages(chat_data):
    """
    We'll only use roles "user" and "assistant" to avoid schema issues.
    1) Possibly force chain-of-thought by prepending a user message if force_thinking = True.
    2) Insert the system prompt as a user message if not empty.
    3) Then add the conversation messages.
    """
    bedrock_msgs = []

    if chat_data.get("force_thinking", False):
        # Force chain-of-thought with a user message
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Please show detailed reasoning in <thinking>...</thinking> if possible."}
            ]
        })

    system_prompt = chat_data.get("system_prompt", "").strip()
    if system_prompt:
        # We'll treat the system prompt as if user said it
        bedrock_msgs.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"(System Instruction)\n{system_prompt}"}
            ]
        })

    for msg in chat_data["messages"]:
        # "role" in { "user", "assistant", "image" (we'll convert "image" to user in the final JSON) }
        if msg["role"] == "image":
            # We'll treat this as a user message with alt text (if any).
            # Because Claude can't handle images natively, we just note it in the conversation.
            alt_text = f"(User uploaded an image): {msg.get('content','[no alt text]')}"
            bedrock_msgs.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": alt_text}
                ]
            })
        else:
            bedrock_msgs.append({
                "role": msg["role"], 
                "content": [
                    {"type": "text", "text": msg["content"]}
                ]
            })
    return bedrock_msgs

# --------------------------------------------------------------------
# BEDROCK INVOKE
# --------------------------------------------------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def invoke_claude(client, chat_data, temperature, max_tokens):
    with st.spinner("Claude is thinking..."):
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
            text_parts = []
            for seg in body["content"]:
                if seg.get("type")=="text":
                    text_parts.append(seg["text"])
            combined = "\n".join(text_parts).strip()
            # Extract <thinking>
            thinking = ""
            match = re.search(r"<thinking>(.*?)</thinking>", combined, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                combined = re.sub(r"<thinking>.*?</thinking>", "", combined, flags=re.DOTALL).strip()
            return combined, thinking
        return "No content from Claude.", ""

# --------------------------------------------------------------------
# CREATE / UPDATE MESSAGE UTILS
# --------------------------------------------------------------------
def create_message(role, content, thinking=""):
    """
    role in { "user", "assistant", "image" }
    """
    return {
        "role": role,
        "content": content,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%I:%M %p")
    }

# --------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------
st.title("Claude Chat (Derek Breese)")

all_chat_names = list(st.session_state.chats.keys())
chat_choice = st.selectbox(
    "Select Conversation",
    options=all_chat_names,
    index=all_chat_names.index(st.session_state.current_chat)
    if st.session_state.current_chat in all_chat_names
    else 0
)
if chat_choice != st.session_state.current_chat:
    st.session_state.current_chat = chat_choice
    st.experimental_rerun()

current_chat = st.session_state.chats[st.session_state.current_chat]

# Create new conversation
new_chat_name = st.text_input("New Chat Name", "")
if st.button("Create Conversation"):
    if new_chat_name and new_chat_name not in st.session_state.chats:
        st.session_state.chats[new_chat_name] = {
            "messages": [],
            "system_prompt": "You are Claude. Provide detailed reasoning in <thinking>...</thinking> if forced.",
            "force_thinking": False
        }
        st.session_state.current_chat = new_chat_name
        st.experimental_rerun()

# Model settings
st.subheader("Model Settings")
temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.slider("Max Tokens", 100, 4096, 1000)

# System Prompt
st.subheader("System Prompt")
current_chat["system_prompt"] = st.text_area(
    "Modify instruction to Claude here.",
    value=current_chat.get("system_prompt","")
)

# Force Thinking Toggle
st.subheader("Chain-of-Thought")
current_chat["force_thinking"] = st.checkbox(
    "Force Thinking in <thinking>...</thinking>?",
    value=current_chat.get("force_thinking",False)
)
st.session_state.show_thinking = st.checkbox(
    "Show Thinking",
    value=st.session_state.show_thinking
)

# Token usage & AWS credits
st.subheader("Usage")
token_count = total_token_usage(current_chat)
st.write(f"Approx. Tokens in this conversation: **{token_count}**")

st.session_state.aws_credits = st.number_input(
    "Estimated AWS Credits Left (USD)",
    value=st.session_state.aws_credits,
    step=1.0
)
# We won't attempt real cost calculations here. You can do something like:
# cost_per_1k_tokens = 0.001  # example
# cost_estimate = (token_count/1000) * cost_per_1k_tokens
# st.write(f"Estimated cost: ${cost_estimate:.3f}")

# S3 Save/Load
st.subheader("Save/Load to S3")
colA, colB = st.columns(2)
with colA:
    if st.button("Save to S3"):
        # current_chat is a dict with messages, system_prompt, force_thinking
        save_chat_to_s3(st.session_state.current_chat, current_chat)
with colB:
    if st.button("Load from S3"):
        loaded = load_chat_from_s3(st.session_state.current_chat)
        if loaded is None:
            st.warning(f"No S3 chat found for '{st.session_state.current_chat}'.")
        else:
            st.session_state.chats[st.session_state.current_chat] = loaded
            st.success(f"Loaded chat '{st.session_state.current_chat}' from S3.")
            st.experimental_rerun()

# Quick list of S3 chats
saved_chats_s3 = list_chats_s3()
if saved_chats_s3:
    chat_to_load = st.selectbox("Load existing S3 chat", ["--Select--"] + saved_chats_s3)
    if chat_to_load != "--Select--":
        if st.button("Load Chosen S3 Chat"):
            loaded_2 = load_chat_from_s3(chat_to_load)
            if loaded_2:
                st.session_state.chats[chat_to_load] = loaded_2
                st.session_state.current_chat = chat_to_load
                st.success(f"Loaded S3 chat '{chat_to_load}'.")
                st.experimental_rerun()

# Clear Chat
if st.button("Clear Current Conversation"):
    current_chat["messages"] = []
    st.experimental_rerun()

# --------------------------------------------------------------------
# MAIN Chat Display
# --------------------------------------------------------------------
for i, msg in enumerate(current_chat["messages"]):
    bubble_class = "assistant-bubble" if msg["role"]=="assistant" else "user-bubble"
    # special case if role == "image"
    if msg["role"] == "image":
        # Display the image
        with st.chat_message("user"):
            st.markdown(f"<div class='image-bubble'>", unsafe_allow_html=True)
            if msg.get("thinking"):
                st.markdown(f"<em>{msg['thinking']}</em><br>", unsafe_allow_html=True)
            # 'content' might store a small alt text or similar
            st.caption(msg.get("content","No alt text"))
            # We might store the base64 data in msg["base64"]
            if "base64" in msg:
                img_data = base64.b64decode(msg["base64"])
                st.image(img_data, caption=msg["timestamp"], use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            # Let user edit alt text or delete
            col1, col2, _ = st.columns([1,1,8])
            with col1:
                if st.button("🗑️ Del", key=f"delImg_{i}"):
                    current_chat["messages"].pop(i)
                    st.experimental_rerun()
    else:
        with st.chat_message(msg["role"]):
            if msg["role"]=="user" and st.session_state.editing_idx == i:
                # Edit mode
                new_text = st.text_area("Edit your message", value=msg["content"], key=f"edit_{i}")
                cA, cB = st.columns([1,1])
                with cA:
                    if st.button("Save", key=f"save_{i}"):
                        current_chat["messages"][i]["content"] = new_text
                        st.session_state.editing_idx = None
                        st.experimental_rerun()
                with cB:
                    if st.button("Cancel", key=f"cancel_{i}"):
                        st.session_state.editing_idx = None
                        st.experimental_rerun()
            else:
                # Normal bubble
                st.markdown(f"<div class='chat-bubble {bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
                st.caption(msg.get("timestamp",""))
                # If assistant and user wants to see chain-of-thought
                if msg["role"]=="assistant" and st.session_state.show_thinking and msg.get("thinking"):
                    st.markdown(f"<div class='thinking-box'>{msg['thinking']}</div>", unsafe_allow_html=True)

                if msg["role"]=="user":
                    # Edit / delete row
                    c1, c2, _ = st.columns([1,1,8])
                    with c1:
                        if st.button("✏️ Edit", key=f"editbtn_{i}"):
                            st.session_state.editing_idx = i
                            st.experimental_rerun()
                    with c2:
                        if st.button("🗑️ Del", key=f"delbtn_{i}"):
                            current_chat["messages"].pop(i)
                            st.experimental_rerun()

# --------------------------------------------------------------------
# BOTTOM Chat Input & Image Uploader
# --------------------------------------------------------------------
col_input, col_image = st.columns([3,1])
with col_input:
    user_input = st.chat_input("Type your message to Claude...")
    if user_input:
        # Add user text message
        msg_obj = create_message("user", user_input)
        current_chat["messages"].append(msg_obj)
        # Call Claude
        client = get_bedrock_client()
        if client:
            resp_text, resp_think = invoke_claude(
                client,
                current_chat,
                temperature=temperature,
                max_tokens=max_tokens
            )
            current_chat["messages"].append(create_message("assistant", resp_text, resp_think))
        st.experimental_rerun()

with col_image:
    uploaded_img = st.file_uploader("Upload an image", type=["png","jpg","jpeg","gif"])
    if uploaded_img is not None:
        # Convert to base64
        img_bytes = uploaded_img.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        # Optionally let user add alt text
        alt_text = st.text_input("Optional alt text for the image", "")
        if st.button("Add Image to Chat"):
            img_msg = create_message(
                "image",
                alt_text if alt_text else "[No alt text]",
                ""
            )
            img_msg["base64"] = img_base64
            current_chat["messages"].append(img_msg)
            st.success("Image added to conversation.")
            st.experimental_rerun()
