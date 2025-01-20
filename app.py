import streamlit as st 
import boto3
import json
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Page config
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional CSS with better mobile support
st.markdown("""
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
    /* Improved mobile styling */
    @media (max-width: 768px) {
        .stButton button {
            padding: 0.5rem !important;
            width: auto !important;
        }
        .row-widget.stButton {
            margin: 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with error checking
def init_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {
            "Default": {
                "messages": [],
                "settings": {"temperature": 0.7, "max_tokens": 1000},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "Default"
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None
    
    # Ensure current_chat exists
    if st.session_state.current_chat not in st.session_state.chats:
        st.session_state.current_chat = "Default"
        if "Default" not in st.session_state.chats:
            st.session_state.chats["Default"] = {
                "messages": [],
                "settings": {"temperature": 0.7, "max_tokens": 1000},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

init_session_state()

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6))
def get_chat_response(message: str, client, settings: dict):
    """Get response from Claude with proper error handling and response parsing"""
    try:
        with st.spinner("Claude is thinking..."):
            response = client.invoke_model(
                modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "top_k": 250,
                    "top_p": 0.999,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": message
                                }
                            ]
                        }
                    ]
                })
            )
            
            response_body = json.loads(response.get('body').read())
            if 'content' in response_body:
                for content in response_body['content']:
                    if content['type'] == 'text':
                        return content['text']
            return "No response generated. Please try again."
            
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None

@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock client with error handling"""
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

def process_message(message: str, role: str) -> dict:
    """Format message with proper error handling and cleaning"""
    try:
        # Clean and extract message if needed
        cleaned_message = message
        if isinstance(message, str):
            try:
                msg_data = json.loads(message)
                if isinstance(msg_data, list) and msg_data and 'text' in msg_data[0]:
                    cleaned_message = msg_data[0]['text']
            except json.JSONDecodeError:
                pass  # Message is already clean
        
        return {
            "role": role,
            "content": cleaned_message,
            "timestamp": datetime.now().strftime('%I:%M %p')
        }
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        return {
            "role": role,
            "content": "Error processing message",
            "timestamp": datetime.now().strftime('%I:%M %p')
        }

# Sidebar with error boundaries
with st.sidebar:
    st.title("Chat Settings")
    
    try:
        # Chat Management
        st.subheader("Conversations")
        new_chat_name = st.text_input("New Chat Name", key="new_chat_input")
        if st.button("Create Chat") and new_chat_name:
            if new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "settings": {"temperature": 0.7, "max_tokens": 1000},
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()
        
        st.session_state.current_chat = st.selectbox(
            "Select Chat",
            options=list(st.session_state.chats.keys()),
            key="chat_selector"
        )
        
        # Model Settings
        st.subheader("Model Settings")
        current_chat = st.session_state.chats[st.session_state.current_chat]
        
        current_chat["settings"]["temperature"] = st.slider(
            "Temperature",
            0.0, 1.0,
            current_chat["settings"]["temperature"],
            help="Higher = more creative, Lower = more focused"
        )
        
        current_chat["settings"]["max_tokens"] = st.slider(
            "Max Tokens",
            100, 4096,
            current_chat["settings"]["max_tokens"],
            help="Maximum length of response"
        )
        
        # Clear Chat with confirmation
        if st.button("Clear Current Chat"):
            if st.session_state.current_chat in st.session_state.chats:
                if len(current_chat["messages"]) > 0:
                    if st.button("Confirm Clear?", key="confirm_clear"):
                        current_chat["messages"] = []
                        st.rerun()
                else:
                    current_chat["messages"] = []
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error in sidebar: {str(e)}")
        init_session_state()  # Reset to safe state

# Main chat interface
st.title(f"Claude Chat - {st.session_state.current_chat}")

try:
    # Message display with error handling
    for idx, message in enumerate(current_chat["messages"]):
        with st.chat_message(message["role"]):
            if st.session_state.editing_message == idx and message["role"] == "user":
                # Edit mode
                edited_message = st.text_area(
                    "Edit message",
                    message["content"],
                    key=f"edit_{idx}"
                )
                cols = st.columns([1,4])
                with cols[0]:
                    if st.button("Save", key=f"save_{idx}"):
                        current_chat["messages"][idx]["content"] = edited_message
                        st.session_state.editing_message = None
                        st.rerun()
                with cols[1]:
                    if st.button("Cancel", key=f"cancel_{idx}"):
                        st.session_state.editing_message = None
                        st.rerun()
            else:
                # Normal message display
                st.markdown(message["content"])
                st.caption(f"Time: {message['timestamp']}")
                
                # Message controls for user messages
                if message["role"] == "user":
                    cols = st.columns([1,1,10])
                    with cols[0]:
                        if st.button("✏️", key=f"edit_{idx}", help="Edit message"):
                            st.session_state.editing_message = idx
                            st.rerun()
                    with cols[1]:
                        if st.button("🗑️", key=f"delete_{idx}", help="Delete message"):
                            current_chat["messages"].pop(idx)
                            st.rerun()

    # Chat input with error handling
    if prompt := st.chat_input("Message Claude...", key="chat_input"):
        # Add user message
        current_chat["messages"].append(process_message(prompt, "user"))
        
        # Get Claude response
        client = get_bedrock_client()
        if client:
            response = get_chat_response(
                prompt,
                client,
                current_chat["settings"]
            )
            
            if response:
                current_chat["messages"].append(
                    process_message(response, "assistant")
                )
                st.rerun()

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    if st.button("Reset Application"):
        init_session_state()
        st.rerun()
