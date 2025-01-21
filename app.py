import streamlit as st
import boto3
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# -----------------------------------------------------------------------------
# CONFIGURATION
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
# CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.thinking-box {
    background-color: #1E1E1E;
    border-left: 3px solid #FFD700;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}
.message-timestamp {
    color: #666;
    font-size: 0.8em;
    margin-top: 5px;
}
.stButton > button {
    width: 100%;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "messages": [],
        "conversations": {},
        "current_conversation": "Default",
        "system_prompt": "You are Claude. Include detailed chain-of-thought when appropriate.",
        "temperature": 0.7,
        "max_tokens": 1000,
        "show_thinking": True,
        "force_thinking": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    # Initialize default conversation if needed
    if "Default" not in st.session_state.conversations:
        st.session_state.conversations["Default"] = {
            "messages": [],
            "system_prompt": st.session_state.system_prompt,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

init_session_state()

# -----------------------------------------------------------------------------
# AWS CLIENTS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_aws_clients():
    """Create and cache AWS clients."""
    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        return bedrock, s3
    except Exception as e:
        st.error(f"Failed to initialize AWS clients: {str(e)}")
        return None, None

# -----------------------------------------------------------------------------
# S3 OPERATIONS
# -----------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_conversation_to_s3(conversation_name: str, data: dict) -> bool:
    """Save conversation data to S3 with retry logic."""
    _, s3 = get_aws_clients()
    if not s3:
        return False
        
    try:
        key = f"conversations/{conversation_name}.json"
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=json.dumps(data, indent=2),
            ContentType="application/json"
        )
        return True
    except ClientError as e:
        st.error(f"Failed to save conversation: {str(e)}")
        return False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def list_s3_conversations() -> List[str]:
    """List available conversations in S3."""
    _, s3 = get_aws_clients()
    if not s3:
        return []
        
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix="conversations/"
        )
        conversations = []
        if "Contents" in response:
            for item in response["Contents"]:
                name = item["Key"].split("/")[-1].replace(".json", "")
                if name:
                    conversations.append(name)
        return sorted(conversations)
    except ClientError as e:
        st.error(f"Failed to list conversations: {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_conversation_from_s3(conversation_name: str) -> Optional[dict]:
    """Load conversation data from S3 with retry logic."""
    _, s3 = get_aws_clients()
    if not s3:
        return None
        
    try:
        key = f"conversations/{conversation_name}.json"
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchKey":
            st.error(f"Failed to load conversation: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# CLAUDE INTERACTION
# -----------------------------------------------------------------------------
def extract_thinking(response: str) -> Tuple[str, str]:
    """Extract thinking process from Claude's response."""
    thinking = ""
    content = response
    
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", response, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        content = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL).strip()
    
    return content, thinking

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_claude_response(messages: List[dict], system_prompt: str, temperature: float, max_tokens: int) -> Tuple[str, str]:
    """Get response from Claude with retry logic."""
    bedrock, _ = get_aws_clients()
    if not bedrock:
        return "Error: Could not connect to Claude.", ""
        
    try:
        # Prepare messages with system prompt
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        if st.session_state.force_thinking:
            formatted_messages.append({
                "role": "user",
                "content": "Please include your chain-of-thought in <thinking> tags."
            })
        
        response = bedrock.invoke_model(
            modelId=MODEL_ARN,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": formatted_messages
            })
        )
        
        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]
        return extract_thinking(content)
        
    except Exception as e:
        st.error(f"Error getting Claude response: {str(e)}")
        return "Error communicating with Claude.", ""

# -----------------------------------------------------------------------------
# UI COMPONENTS
# -----------------------------------------------------------------------------
def render_message(message: dict):
    """Render a single message with timestamp and thinking process."""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show timestamp
        st.markdown(
            f"<div class='message-timestamp'>{message.get('timestamp', '')}</div>",
            unsafe_allow_html=True
        )
        
        # Show thinking process if available and enabled
        if (message.get("thinking") and 
            message["role"] == "assistant" and 
            st.session_state.show_thinking):
            with st.expander("💭 Thought Process"):
                st.markdown(
                    f"<div class='thinking-box'>{message['thinking']}</div>",
                    unsafe_allow_html=True
                )

def get_current_conversation() -> dict:
    """Get the current conversation data."""
    return st.session_state.conversations[st.session_state.current_conversation]

def create_new_conversation(name: str):
    """Create a new conversation."""
    if name and name not in st.session_state.conversations:
        st.session_state.conversations[name] = {
            "messages": [],
            "system_prompt": st.session_state.system_prompt,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.current_conversation = name

# -----------------------------------------------------------------------------
# MAIN UI LAYOUT
# -----------------------------------------------------------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.title("🛠️ Chat Settings")
        
        # Model settings
        st.subheader("Model Configuration")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1
        )
        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=st.session_state.max_tokens,
            step=100
        )
        
        # Thinking process controls
        st.subheader("Chain of Thought")
        st.session_state.show_thinking = st.checkbox(
            "Show Thinking Process",
            value=st.session_state.show_thinking
        )
        st.session_state.force_thinking = st.checkbox(
            "Force Thinking in Responses",
            value=st.session_state.force_thinking
        )
        
        # System prompt
        st.subheader("System Prompt")
        new_prompt = st.text_area(
            "System Instructions",
            value=get_current_conversation()["system_prompt"],
            height=100
        )
        if new_prompt != get_current_conversation()["system_prompt"]:
            get_current_conversation()["system_prompt"] = new_prompt
        
        # Conversation management
        st.subheader("Conversation Management")
        new_chat_name = st.text_input("New Conversation Name")
        if st.button("Create New Conversation"):
            create_new_conversation(new_chat_name)
        
        # S3 operations
        st.subheader("S3 Operations")
        if st.button("Save Current Conversation"):
            if save_conversation_to_s3(
                st.session_state.current_conversation,
                get_current_conversation()
            ):
                st.success("Conversation saved successfully!")
        
        s3_conversations = list_s3_conversations()
        if s3_conversations:
            selected_s3_chat = st.selectbox(
                "Load Conversation from S3",
                ["Select..."] + s3_conversations
            )
            if selected_s3_chat != "Select..." and st.button("Load Selected"):
                data = load_conversation_from_s3(selected_s3_chat)
                if data:
                    st.session_state.conversations[selected_s3_chat] = data
                    st.session_state.current_conversation = selected_s3_chat
                    st.success(f"Loaded '{selected_s3_chat}' successfully!")
                    st.rerun()
        
        if st.button("Clear Current Conversation"):
            get_current_conversation()["messages"] = []
            st.rerun()

    # Main chat area
    st.title(f"💬 {st.session_state.current_conversation}")
    
    # Display conversation selector
    st.session_state.current_conversation = st.selectbox(
        "Select Conversation",
        options=list(st.session_state.conversations.keys()),
        index=list(st.session_state.conversations.keys()).index(
            st.session_state.current_conversation
        )
    )
    
    # Display messages
    for message in get_current_conversation()["messages"]:
        render_message(message)
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
        get_current_conversation()["messages"].append(user_message)
        
        # Get Claude's response
        content, thinking = get_claude_response(
            get_current_conversation()["messages"],
            get_current_conversation()["system_prompt"],
            st.session_state.temperature,
            st.session_state.max_tokens
        )
        
        # Add assistant message
        assistant_message = {
            "role": "assistant",
            "content": content,
            "thinking": thinking,
            "timestamp": datetime.now().strftime("%I:%M %p")
        }
        get_current_conversation()["messages"].append(assistant_message)
        
        # Rerun to update display
        st.rerun()

if __name__ == "__main__":
    main()
