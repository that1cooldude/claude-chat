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

# Enhanced UI styling
st.markdown("""
<style>
    /* Message Bubbles */
    .message-container {
        margin: 15px 0;
        padding: 10px;
        border-radius: 15px;
        position: relative;
    }
    
    .user-message {
        background-color: #2e3136;
        margin-left: 20px;
        border: 1px solid #404040;
    }
    
    .assistant-message {
        background-color: #36393f;
        margin-right: 20px;
        border: 1px solid #404040;
    }
    
    /* Thinking Process */
    .thinking-container {
        background-color: #1e1e2e;
        border-left: 3px solid #ffd700;
        padding: 10px;
        margin: 10px 0;
        font-style: italic;
        color: #b8b8b8;
    }
    
    /* Code Blocks */
    .stCodeBlock {
        background-color: #1e1e1e !important;
        padding: 1em;
        border-radius: 5px;
        position: relative;
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
        text-align: right;
    }
    
    /* Chat Selection */
    .chat-selector {
        margin-bottom: 20px;
    }
    
    /* System Prompt */
    .system-prompt {
        background-color: #2a2d2e;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .message-container {
            margin: 10px 5px;
        }
        .stButton>button {
            width: 100%;
        }
        .stTextInput>div>div>input {
            font-size: 16px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": "You are a helpful AI assistant. Always show your reasoning using <thinking></thinking> tags before providing your final response."
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default"
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = {}
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# Initialize Bedrock client
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
    
    # Chat Creation
    cols = st.columns([2, 1])
    with cols[0]:
        new_chat_name = st.text_input("New Chat Name")
    with cols[1]:
        if st.button("Create") and new_chat_name:
            if new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": st.session_state.chats[st.session_state.current_chat]["system_prompt"]
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()
    
    # Chat Selection
    st.session_state.current_chat = st.selectbox(
        "Select Chat",
        options=list(st.session_state.chats.keys()),
        index=list(st.session_state.chats.keys()).index(st.session_state.current_chat)
    )
    
    # Delete Chat
    if len(st.session_state.chats) > 1 and st.button("Delete Current Chat"):
        del st.session_state.chats[st.session_state.current_chat]
        st.session_state.current_chat = list(st.session_state.chats.keys())[0]
        st.rerun()
    
    st.divider()
    
    # System Prompt
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "Customize System Prompt",
        value=st.session_state.chats[st.session_state.current_chat]["system_prompt"],
        help="Define how Claude should behave"
    )
    
    # Quick System Prompts
    st.subheader("Quick Prompts")
    if st.button("Data Scientist"):
        system_prompt = """You are a data science assistant with expertise in statistics, machine learning, and programming.
        Always show your reasoning process using <thinking></thinking> tags.
        When sharing code, include detailed comments and explanations."""
        st.session_state.chats[st.session_state.current_chat]["system_prompt"] = system_prompt
        st.rerun()
    
    if st.button("Code Helper"):
        system_prompt = """You are a coding assistant focused on helping with programming tasks.
        Always show your reasoning using <thinking></thinking> tags.
        Provide well-commented code examples and step-by-step explanations."""
        st.session_state.chats[st.session_state.current_chat]["system_prompt"] = system_prompt
        st.rerun()
    
    if system_prompt != st.session_state.chats[st.session_state.current_chat]["system_prompt"]:
        st.session_state.chats[st.session_state.current_chat]["system_prompt"] = system_prompt
    
    st.divider()
    
    # UI Settings
    st.subheader("Settings")
    st.toggle("Show Thinking Process", key="show_thinking")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1000)
    
    # Prompt Cache
    st.subheader("Saved Prompts")
    with st.expander("Manage Prompts"):
        for name, saved_prompt in st.session_state.prompt_cache.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{name}: {saved_prompt[:50]}...")
            with col2:
                if st.button("Use", key=f"use_{name}"):
                    st.session_state.reuse_prompt = saved_prompt
                    st.rerun()
    
    # Stats
    st.divider()
    st.subheader("Session Stats")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    if st.button("Clear Current Chat"):
        st.session_state.chats[st.session_state.current_chat]["messages"] = []
        st.rerun()

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Display messages
for idx, message in enumerate(st.session_state.chats[st.session_state.current_chat]["messages"]):
    with st.chat_message(message["role"]):
        st.markdown(f"""
        <div class="message-container {message['role']}-message">
            {message['content']}
            <div class="timestamp">{message.get('timestamp', datetime.now().strftime('%I:%M %p'))}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and 'thinking' in message:
            with st.expander("View Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(f"""
                <div class="thinking-container">
                    {message['thinking']}
                </div>
                """, unsafe_allow_html=True)

        # Save prompt option
        if message['role'] == 'user':
            with st.expander("Save this prompt"):
                prompt_name = st.text_input("Name", key=f"save_prompt_{idx}")
                if st.button("Save", key=f"save_button_{idx}"):
                    if prompt_name:
                        st.session_state.prompt_cache[prompt_name] = message['content']
                        st.success(f"Saved as {prompt_name}")

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    timestamp = datetime.now().strftime('%I:%M %p')
    st.session_state.chats[st.session_state.current_chat]["messages"].append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
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
                # Force thinking tags in prompt
                enhanced_prompt = f"""Please approach this request step by step, showing your reasoning process within <thinking></thinking> tags before providing your final response.

{prompt}

Remember to structure your response with <thinking> tags first, then your final answer."""

                response = client.invoke_model(
                    modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [
                            {
                                "role": "system",
                                "content": st.session_state.chats[st.session_state.current_chat]["system_prompt"]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": enhanced_prompt}]
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
                    with st.expander("View Thinking Process", expanded=True):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {thinking_process}
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Export button for current chat
if st.session_state.chats[st.session_state.current_chat]["messages"]:
    st.sidebar.divider()
    if st.sidebar.button("Export Current Chat"):
        chat_text = f"# Chat Export: {st.session_state.current_chat}\n\n"
        chat_text += f"System Prompt: {st.session_state.chats[st.session_state.current_chat]['system_prompt']}\n\n"
        for msg in st.session_state.chats[st.session_state.current_chat]["messages"]:
            chat_text += f"## {msg['role'].title()} ({msg.get('timestamp', 'N/A')})\n"
            chat_text += f"{msg['content']}\n\n"
            if msg['role'] == 'assistant' and 'thinking' in msg:
                chat_text += f"### Thinking Process\n{msg['thinking']}\n\n"
        
        st.sidebar.download_button(
            "Download Chat",
            chat_text,
            file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
