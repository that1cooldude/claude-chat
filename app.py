import streamlit as st
import boto3
import json
import os
from datetime import datetime
import time
import re

# Page config
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .message-container { margin: 15px 0; padding: 10px; border-radius: 15px; }
    .user-message { background-color: #2e3136; margin-left: 20px; border: 1px solid #404040; }
    .assistant-message { background-color: #36393f; margin-right: 20px; border: 1px solid #404040; }
    .thinking-container { 
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 10px;
        margin: 10px 0;
        font-style: italic;
        color: #b8b8b8;
    }
    .timestamp { font-size: 0.8em; color: #666; text-align: right; }
    .system-prompt { 
        background-color: #2a2d2e;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .chat-stats {
        font-size: 0.8em;
        color: #888;
        margin-top: 5px;
    }
    code { padding: 2px 5px; background: #1e1e1e; border-radius: 3px; }
    .copyable { position: relative; }
    .copy-button {
        position: absolute;
        top: 5px;
        right: 5px;
        padding: 3px 6px;
        background: #404040;
        border: none;
        border-radius: 3px;
        color: white;
        cursor: pointer;
    }
    @media (max-width: 768px) {
        .message-container { margin: 10px 5px; }
        .stButton>button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": "You are a helpful AI assistant. Always show your reasoning using <thinking></thinking> tags before providing your final response.",
            "settings": {
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default"
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# Bedrock client initialization - keeping working inference setup
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=st.secrets["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

# Sidebar
with st.sidebar:
    st.title("Chat Management")
    
    # Create new chat
    new_chat_name = st.text_input("New Chat Name")
    col1, col2 = st.columns([2,1])
    with col1:
        template = st.selectbox("Template", ["Default", "Data Scientist", "Code Helper"])
    with col2:
        if st.button("Create") and new_chat_name:
            if new_chat_name not in st.session_state.chats:
                template_prompts = {
                    "Default": "You are a helpful AI assistant. Always show your reasoning using <thinking></thinking> tags.",
                    "Data Scientist": "You are a data science expert. Show your reasoning with <thinking></thinking> tags. Include code examples with detailed comments.",
                    "Code Helper": "You are a coding assistant. Break down problems step by step using <thinking></thinking> tags. Provide well-commented code examples."
                }
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": template_prompts[template],
                    "settings": {
                        "temperature": 0.7,
                        "max_tokens": 1000
                    },
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()
    
    # Chat selection
    st.divider()
    current_chat = st.selectbox(
        "Select Chat",
        options=list(st.session_state.chats.keys()),
        index=list(st.session_state.chats.keys()).index(st.session_state.current_chat)
    )
    if current_chat != st.session_state.current_chat:
        st.session_state.current_chat = current_chat
        st.rerun()
    
    # System prompt
    st.divider()
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Customize",
        value=st.session_state.chats[st.session_state.current_chat]["system_prompt"]
    )
    if system_prompt != st.session_state.chats[st.session_state.current_chat]["system_prompt"]:
        st.session_state.chats[st.session_state.current_chat]["system_prompt"] = system_prompt
    
    # Settings
    st.divider()
    st.subheader("Settings")
    show_thinking = st.toggle("Show Thinking Process", value=st.session_state.show_thinking)
    if show_thinking != st.session_state.show_thinking:
        st.session_state.show_thinking = show_thinking
    
    temperature = st.slider(
        "Temperature",
        0.0, 1.0,
        st.session_state.chats[st.session_state.current_chat]["settings"]["temperature"]
    )
    max_tokens = st.slider(
        "Max Tokens",
        100, 4096,
        st.session_state.chats[st.session_state.current_chat]["settings"]["max_tokens"]
    )
    
    # Update settings if changed
    current_settings = st.session_state.chats[st.session_state.current_chat]["settings"]
    if temperature != current_settings["temperature"] or max_tokens != current_settings["max_tokens"]:
        st.session_state.chats[st.session_state.current_chat]["settings"].update({
            "temperature": temperature,
            "max_tokens": max_tokens
        })
    
    # Chat management buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Chat"):
            chat = st.session_state.chats[st.session_state.current_chat]
            export_data = {
                "chat_name": st.session_state.current_chat,
                "created_at": chat["created_at"],
                "system_prompt": chat["system_prompt"],
                "messages": chat["messages"],
                "settings": chat["settings"]
            }
            st.download_button(
                "Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chats[st.session_state.current_chat]["messages"] = []
            st.rerun()
    
    if len(st.session_state.chats) > 1:
        if st.button("Delete Chat"):
            del st.session_state.chats[st.session_state.current_chat]
            st.session_state.current_chat = list(st.session_state.chats.keys())[0]
            st.rerun()
    
    # Stats
    st.divider()
    st.subheader("Session Stats")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Display messages
for idx, message in enumerate(st.session_state.chats[st.session_state.current_chat]["messages"]):
    with st.chat_message(message["role"]):
        # Message container with timestamp
        st.markdown(f"""
        <div class="message-container {message['role']}-message">
            {message['content']}
            <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display thinking process if available
        if message['role'] == 'assistant' and 'thinking' in message:
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(f"""
                <div class="thinking-container">
                    {message['thinking']}
                </div>
                """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    timestamp = datetime.now().strftime('%I:%M %p')
    st.session_state.chats[st.session_state.current_chat]["messages"].append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            {prompt}
            <div class="timestamp">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Claude's response
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        try:
            with st.spinner("Thinking..."):
                # Include system prompt with user message
                current_chat = st.session_state.chats[st.session_state.current_chat]
                enhanced_prompt = f"""Please approach this request step by step, showing your reasoning process within <thinking></thinking> tags before providing your final response.

{prompt}

Remember to structure your response with <thinking> tags first, then your final answer."""

                # The working inference setup
                response = client.invoke_model(
                    modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": current_chat["settings"]["max_tokens"],
                        "temperature": current_chat["settings"]["temperature"],
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{current_chat['system_prompt']}\n\n{enhanced_prompt}"
                                    }
                                ]
                            }
                        ]
                    })
                )
                
                response_body = json.loads(response['body'].read())
                full_response = response_body['content'][0]['text']
                
                # Extract thinking and main response
                thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
                thinking_process = thinking_match.group(1) if thinking_match else ""
                main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
                
                # Store response
                timestamp = datetime.now().strftime('%I:%M %p')
                st.session_state.chats[st.session_state.current_chat]["messages"].append({
                    "role": "assistant",
                    "content": main_response,
                    "thinking": thinking_process,
                    "timestamp": timestamp
                })
                
                # Display response
                st.markdown(f"""
                <div class="message-container assistant-message">
                    {main_response}
                    <div class="timestamp">{timestamp}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if thinking_process and st.session_state.show_thinking:
                    with st.expander("Thinking Process", expanded=True):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {thinking_process}
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("If you're seeing an authentication error, please check your AWS credentials.")
