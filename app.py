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

# Clean, professional CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid;
    }
    .user-message {
        background-color: #1e1e2e;
        border-left-color: #7289da;
    }
    .assistant-message {
        background-color: #262633;
        border-left-color: #43b581;
    }
    .message-timestamp {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
    }
    .edit-button, .delete-button {
        background-color: transparent;
        border: 1px solid #666;
        color: #fff;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0 0.2rem;
        cursor: pointer;
    }
    .edit-button:hover, .delete-button:hover {
        background-color: rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {
        "Default": {
            "messages": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = "Default"
if "editing" not in st.session_state:
    st.session_state.editing = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_claude_response(client, prompt: str, max_tokens: int = 1000) -> str:
    """Get response from Claude with proper error handling"""
    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens": max_tokens,
                "temperature": 0.7
            })
        )
        return json.loads(response.get('body').read()).get('completion', '')
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None

def get_bedrock_client():
    """Initialize Bedrock client with proper error handling"""
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

# Sidebar
with st.sidebar:
    st.title("Chat Settings")
    
    # Conversation Management
    st.subheader("Conversations")
    new_chat_name = st.text_input("New Conversation Name")
    if st.button("Create") and new_chat_name:
        if new_chat_name not in st.session_state.conversations:
            st.session_state.conversations[new_chat_name] = {
                "messages": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.current_conversation = new_chat_name
            st.rerun()
    
    st.session_state.current_conversation = st.selectbox(
        "Select Conversation",
        options=list(st.session_state.conversations.keys())
    )
    
    if st.button("Clear Current Conversation"):
        if st.session_state.current_conversation in st.session_state.conversations:
            st.session_state.conversations[st.session_state.current_conversation]["messages"] = []
            st.rerun()

# Main chat interface
st.title(f"Claude Chat - {st.session_state.current_conversation}")

current_chat = st.session_state.conversations[st.session_state.current_conversation]

# Display messages
for idx, message in enumerate(current_chat["messages"]):
    # Handle message editing
    if st.session_state.editing == idx:
        with st.container():
            edited_message = st.text_area(
                "Edit message",
                value=message["content"],
                key=f"edit_{idx}"
            )
            col1, col2 = st.columns([1,4])
            with col1:
                if st.button("Save", key=f"save_{idx}"):
                    current_chat["messages"][idx]["content"] = edited_message
                    st.session_state.editing = None
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_{idx}"):
                    st.session_state.editing = None
                    st.rerun()
    else:
        # Normal message display
        message_class = "user-message" if message["role"] == "user" else "assistant-message"
        
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="message-content">{message["content"]}</div>
                <div class="message-timestamp">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Message controls
            if message["role"] == "user":
                col1, col2 = st.columns([1,20])
                with col1:
                    if st.button("✏️", key=f"edit_{idx}"):
                        st.session_state.editing = idx
                        st.rerun()
                with col1:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        current_chat["messages"].pop(idx)
                        st.rerun()

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    current_chat["messages"].append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })
    
    # Get Claude response
    client = get_bedrock_client()
    if client:
        with st.spinner("Claude is thinking..."):
            response = get_claude_response(client, prompt)
            if response:
                current_chat["messages"].append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
