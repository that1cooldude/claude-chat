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
    page_title="Enhanced Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CSS
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
div[data-testid="stExpander"] {
    border: none;
    box-shadow: none;
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
    if "show_thinking" not in st.session_state:
        st.session_state.show_thinking = True
    if "processing_message" not in st.session_state:
        st.session_state.processing_message = False
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

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
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_chat_to_s3(chat_name: str, chat_data: dict) -> bool:
    s3 = get_s3_client()
    if not s3:
        return False
    try:
        key = f"conversations/{chat_name}.json"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(chat_data, indent=2),
            ContentType="application/json"
        )
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False

def load_chat_from_s3(chat_name: str) -> dict:
    s3 = get_s3_client()
    if not s3:
        return None
    try:
        key = f"conversations/{chat_name}.json"
        resp = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            st.error(f"Load error: {e}")
        return None

@st.cache_data(ttl=300)
def list_s3_chats() -> list:
    s3 = get_s3_client()
    if not s3:
        return []
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="conversations/")
        if "Contents" not in resp:
            return []
        return sorted([
            k["Key"].split("/")[-1].replace(".json","")
            for k in resp["Contents"]
            if k["Key"].endswith(".json")
        ])
    except Exception as e:
        st.error(f"List error: {e}")
        return []

# -----------------------------------------------------------------------------
# CLAUDE OPERATIONS
# -----------------------------------------------------------------------------
def build_messages(chat_data: dict) -> list:
    messages = []
    
    if chat_data.get("force_thinking", False):
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Please include chain-of-thought in <thinking>...</thinking> tags."}]
        })
    
    sys_prompt = chat_data.get("system_prompt","").strip()
    if sys_prompt:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": f"(System Prompt)\n{sys_prompt}"}]
        })
    
    for msg in chat_data["messages"]:
        messages.append({
            "role": msg["role"],
            "content": [{"type": "text", "text": msg["content"]}]
        })
    
    return messages

@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=6))
def get_claude_response(chat_data: dict, temperature: float, max_tokens: int) -> tuple:
    client = get_bedrock_client()
    if not client:
        raise Exception("Failed to initialize Bedrock client")
    
    messages = build_messages(chat_data)
    try:
        response = client.invoke_model(
            modelId=MODEL_ARN,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages
            })
        )
        
        result = json.loads(response["body"].read())
        
        if "content" not in result:
            raise Exception("Invalid response format from Claude")
            
        # Combine all text segments
        full_response = " ".join(
            seg["text"] for seg in result["content"]
            if seg.get("type") == "text"
        ).strip()
        
        # Extract thinking process
        thinking = ""
        match = re.search(r"<thinking>(.*?)</thinking>", full_response, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            full_response = re.sub(r"<thinking>.*?</thinking>", "", full_response, flags=re.DOTALL).strip()
            
        return full_response, thinking
        
    except Exception as e:
        st.error(f"Error getting Claude response: {str(e)}")
        raise

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
def main():
    col_chat, col_settings = st.columns([2,1], gap="large")
    
    # Settings Column
    with col_settings:
        st.title("Chat Settings")
        
        # Conversation Management
        st.subheader("Conversation")
        chat_keys = list(st.session_state.chats.keys())
        chosen_chat = st.selectbox(
            "Select Conversation",
            options=chat_keys,
            index=chat_keys.index(st.session_state.current_chat)
        )
        if chosen_chat != st.session_state.current_chat:
            st.session_state.current_chat = chosen_chat
            
        # Create New Chat
        new_chat_name = st.text_input("New Chat Name")
        if st.button("Create Chat") and new_chat_name:
            if new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                    "force_thinking": True
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()
                
        chat_data = get_current_chat_data()
        
        # Model Settings
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_tokens = st.slider("Max Tokens", 100, 4096, 1000)
        
        # System Prompt
        st.subheader("System Prompt")
        chat_data["system_prompt"] = st.text_area(
            "Instructions for Claude",
            value=chat_data.get("system_prompt","")
        )
        
        # Thinking Settings
        st.subheader("Chain of Thought")
        chat_data["force_thinking"] = st.checkbox(
            "Request Thinking Process",
            value=chat_data.get("force_thinking", True)
        )
        st.session_state.show_thinking = st.checkbox(
            "Show Thinking Process",
            value=st.session_state.show_thinking
        )
        
        # S3 Operations
        st.subheader("Save/Load")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Chat"):
                if save_chat_to_s3(st.session_state.current_chat, chat_data):
                    st.success("Chat saved!")
                    
        with col2:
            if st.button("Load Chat"):
                data = load_chat_from_s3(st.session_state.current_chat)
                if data:
                    st.session_state.chats[st.session_state.current_chat] = data
                    st.success("Chat loaded!")
                    st.rerun()
                else:
                    st.warning("No saved chat found.")
                    
        # S3 Chat List
        saved_chats = list_s3_chats()
        if saved_chats:
            selected = st.selectbox("Load Saved Chat", ["Select..."] + saved_chats)
            if selected != "Select..." and st.button("Load Selected"):
                data = load_chat_from_s3(selected)
                if data:
                    st.session_state.chats[selected] = data
                    st.session_state.current_chat = selected
                    st.success(f"Loaded '{selected}'")
                    st.rerun()
        
        # Clear Chat
        if st.button("Clear Chat"):
            chat_data["messages"] = []
            st.rerun()
    
    # Chat Column
    with col_chat:
        st.title(f"💬 {st.session_state.current_chat}")
        
        # Display Messages
        for msg in chat_data["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                st.caption(f"Sent at {msg.get('timestamp', '')}")
                
                if (msg["role"] == "assistant" and 
                    st.session_state.show_thinking and 
                    msg.get("thinking")):
                    with st.expander("💭 Thinking Process"):
                        st.markdown(
                            f"<div class='thinking-expander'>"
                            f"<div class='thinking-text'>{msg['thinking']}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
        
        # Chat Input
        if prompt := st.chat_input("Type your message here..."):
            if not st.session_state.processing_message:
                st.session_state.processing_message = True
                
                try:
                    # Add user message
                    chat_data["messages"].append({
                        "role": "user",
                        "content": prompt,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    # Get Claude's response
                    response, thinking = get_claude_response(
                        chat_data,
                        temperature,
                        max_tokens
                    )
                    
                    # Add assistant message
                    chat_data["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "thinking": thinking,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    st.session_state.error_count = 0
                    
                except Exception as e:
                    st.session_state.error_count += 1
                    if st.session_state.error_count >= 3:
                        st.error("Multiple errors occurred. Please check your settings and try again later.")
                    
                finally:
                    st.session_state.processing_message = False
                    st.rerun()

if __name__ == "__main__":
    main()
