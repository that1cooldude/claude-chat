# imports
import streamlit as st
import boto3
import json
import re
from datetime import datetime
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

# CSS styling
st.markdown("""
<style>
/* Primary theming */
:root {
    --primary-color: #2d3748;
    --secondary-color: #3aa1da;
    --background-color: #f5f7fa;
    --text-color: #2d3748;
    --gradient: linear-gradient(135deg, #2d3748 0%, #3aa1da 100%);
    --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* General styling */
body {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Chat container styling */
.chat-container {
    background: var(--primary-color);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: var(--shadow);
}

/* Message styling */
.user-bubble {
    background-color: #3aa1da;
    border-left: 4px solid #ffd700;
    border-radius: 10px;
    padding: 1rem;
    max-width: 80%;
    margin: 0.5rem 0;
    word-wrap: break-word;
}

.assistant-bubble {
    background-color: #2d3748;
    border-right: 4px solid #ffd700;
    border-radius: 10px;
    padding: 1rem;
    max-width: 80%;
    margin: 0.5rem 0;
    word-wrap: break-word;
}

.timestamp {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.6);
    margin-top: 0.5rem;
    text-align: right;
}

/* Thinking expander styling */
.thinking-expander {
    background-color: rgba(0, 0, 0, 0.1);
    border-left: 4px solid #ffd700;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: transform 0.2s ease;
}

.thinking-expander:hover {
    transform: translateX(4px);
}

/* Input and button styling */
.stChatMessage {
    margin: 0;
    padding: 0;
}

.stButton>button {
    background-image: var(--gradient);
    border: none;
    border-radius: 8px;
    color: white;
    padding: 0.8rem 1.5rem;
    font-weight: 500;
    cursor: pointer;
    transition: transform 0.15s ease;
}

.stButton>button:hover {
    transform: translateY(-1px);
}

/* Expander styling */
div[data-testid="stExpander"] {
    border: none;
    box-shadow: none;
    background-color: transparent;
}

/* Improved text display */
.chat-message {
    white-space: pre-wrap;
    font-size: 1rem;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# session state initialization
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

# AWS clients
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            "bedrock-runtime",
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
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
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Error creating S3 client: {e}")
        return None

# S3 operations
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def save_chat_to_s3(chat_name: str, chat_data: dict) -> bool:
    s3 = get_s3_client()
    if not s3:
        return False
    try:
        key = f"conversations/{chat_name}.json"
        s3.put_object(
            Bucket=st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket"),
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
        resp = s3.get_object(Bucket=st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket"), Key=key)
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
        resp = s3.list_objects_v2(Bucket=st.secrets.get("S3_BUCKET_NAME", "my-llm-chats-bucket"), Prefix="conversations/")
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

# Claude operations
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
            modelId=st.secrets.get("MODEL_ARN", 
                "arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
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

# main application
def main():
    # page config
    st.set_page_config(
        page_title="Enhanced Claude Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # columns
    col_chat, col_settings = st.columns([2, 1], gap="large")

    # settings column
    with col_settings:
        st.title("Chat Settings", anchor=False)
        
        # conversation management
        chat_keys = list(st.session_state.chats.keys())
        chosen_chat = st.selectbox(
            "Active Conversation",
            options=chat_keys,
            index=chat_keys.index(st.session_state.current_chat),
            help="Select the conversation you want to work with."
        )
        if chosen_chat != st.session_state.current_chat:
            st.session_state.current_chat = chosen_chat

        # create new chat
        new_chat_name = st.text_input(
            "New Chat Name",
            help="Enter a name for your new conversation."
        )
        if st.button("Create Chat", help="Create a new conversation with the specified name."):
            if new_chat_name and new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": "You are Claude. Provide chain-of-thought if forced.",
                    "force_thinking": True
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()

        chat_data = get_current_chat_data()

        # model settings
        st.subheader("Model Settings", anchor=False)
        temperature = st.slider(
            "Temperature (creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Controls randomness in responses. Lower values make responses more focused."
        )
        max_tokens = st.slider(
            "Maximum Response Length",
            min_value=100,
            max_value=4096,
            value=1000,
            help="Controls the maximum length of Claude's response."
        )

        # system prompt
        st.subheader("System Instructions", anchor=False)
        system_prompt = st.text_area(
            "Claude's Instructions",
            value=chat_data.get("system_prompt", "").strip(),
            height=150,
            help="Define Claude's behavior using system prompts."
        )
        chat_data["system_prompt"] = system_prompt.strip()

        # thinking settings
        st.subheader("Chain of Thought", anchor=False)
        chat_data["force_thinking"] = st.checkbox(
            "Request Thinking Process",
            value=chat_data.get("force_thinking", True),
            help="Enable to request Claude's internal reasoning process."
        )
        st.session_state.show_thinking = st.checkbox(
            "Show Thinking Process",
            value=st.session_state.show_thinking,
            help="Toggle to display Claude's internal thinking steps."
        )

        # S3 operations
        st.subheader("Save/Load Conversations", anchor=False)
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Current Chat", help="Save the current conversation to S3."):
                if save_chat_to_s3(st.session_state.current_chat, chat_data):
                    st.success("Chat saved successfully!", icon="✅")

        with col2:
            if st.button("📤 Load Current Chat", help="Load the current conversation from S3."):
                data = load_chat_from_s3(st.session_state.current_chat)
                if data:
                    st.session_state.chats[st.session_state.current_chat] = data
                    st.success("Chat loaded successfully!", icon="✅")
                    st.rerun()
                else:
                    st.warning("No saved chat found.", icon="⚠️")

        # select and load from saved chats
        saved_chats = list_s_s3_chats()
        if saved_chats:
            selected = st.selectbox(
                "Select Saved Chat",
                ["Select..."] + saved_chats,
                help="Choose a saved conversation to load."
            )
            if selected != "Select..." and st.button("🔄 Load Selected Chat"):
                data = load_chat_from_s3(selected)
                if data:
                    st.session_state.chats[selected] = data
                    st.session_state.current_chat = selected
                    st.success(f"Successfully loaded '{selected}'!", icon="✅")
                    st.rerun()
                else:
                    st.warning("Failed to load the selected chat.", icon="⚠️")

        # clear chat
        if st.button("🗑️ Clear Current Chat", help="Clear all messages in the current conversation."):
            chat_data["messages"] = []
            st.success("Chat cleared!", icon="✅")
            st.rerun()

    # chat column
    with col_chat:
        st.title(f"💬 {st.session_state.current_chat}", anchor=False)

        # display messages
        for msg in chat_data["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                st.caption(f"🕒 Sent at {msg.get('timestamp', '')}")

                if msg["role"] == "assistant" and \
                   st.session_state.show_thinking and \
                   msg.get("thinking"):
                    with st.expander("🧠 Thinking Process", expanded=True):
                        st.markdown(
                            f"<div class='thinking-expander'>"
                            f"<div class='thinking-text'>{msg['thinking']}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

        # chat input
        if prompt := st.chat_input("Type your message here..."):
            if not st.session_state.processing_message:
                st.session_state.processing_message = True
                try:
                    # add user message
                    chat_data["messages"].append({
                        "role": "user",
                        "content": prompt,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    
                    # get response
                    response, thinking = get_claude_response(chat_data, temperature, max_tokens)

                    # add assistant message
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
                        st.error("Multiple errors occurred. Please check your settings and try again.", icon="❌")
                finally:
                    st.session_state.processing_message = False
                    st.rerun()

if __name__ == "__main__":
    main()
