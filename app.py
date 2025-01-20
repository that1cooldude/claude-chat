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
    
    /* Copy Button */
    .copy-button {
        position: absolute;
        top: 5px;
        right: 5px;
        padding: 5px 10px;
        background: #404040;
        border: none;
        border-radius: 3px;
        color: white;
        cursor: pointer;
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
        text-align: right;
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .message-container {
            margin: 10px 5px;
        }
        .stButton>button {
            width: 100%;
        }
    }
    
    /* Typing Indicator */
    .typing-indicator {
        background-color: #404040;
        padding: 5px 10px;
        border-radius: 10px;
        display: inline-block;
        margin: 10px 0;
    }
    
    /* Feedback Buttons */
    .feedback-container {
        display: flex;
        gap: 10px;
        margin-top: 5px;
    }
    
    .feedback-button {
        background: transparent;
        border: 1px solid #666;
        border-radius: 3px;
        padding: 2px 8px;
        color: #666;
        cursor: pointer;
    }
</style>

<script>
function copyText(elementId) {
    const element = document.getElementById(elementId);
    const text = element.textContent;
    navigator.clipboard.writeText(text);
    
    // Show copied indicator
    const button = element.nextElementSibling;
    const originalText = button.textContent;
    button.textContent = 'Copied!';
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
}
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "conversation_name" not in st.session_state:
    st.session_state.conversation_name = f"Conversation_{datetime.now().strftime('%Y%m%d_%H%M')}"
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# Bedrock client (keeping your working version)
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=st.secrets["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

# Enhanced sidebar with UI controls
with st.sidebar:
    st.title("Chat Settings")
    
    # Conversation Management
    st.subheader("Conversation")
    st.text_input("Conversation Name", key="conversation_name")
    
    # UI Settings
    st.subheader("UI Settings")
    st.toggle("Show Thinking Process", key="show_thinking")
    theme = st.selectbox("Theme", ["Dark", "Light"], 
                        index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = theme.lower()
    
    # Model Settings
    st.subheader("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1000)
    
    # Conversation Management
    if st.button("Export Chat"):
        chat_text = "\n\n".join([
            f"**{msg['role']}** ({msg.get('timestamp', 'N/A')}):\n{msg['content']}"
            for msg in st.session_state.messages
        ])
        st.download_button(
            "Download",
            chat_text,
            file_name=f"{st.session_state.conversation_name}.md",
            mime="text/markdown"
        )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()
    
    # Stats
    st.subheader("Session Stats")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")

# Main chat interface
st.title("🤖 Claude Chat")

# Display messages with enhanced UI
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # Message container
        st.markdown(f"""
        <div class="message-container {message['role']}-message">
            {message['content']}
            <div class="timestamp">{message.get('timestamp', datetime.now().strftime('%I:%M %p'))}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display thinking process for assistant messages
        if message['role'] == 'assistant' and 'thinking' in message:
            with st.expander("View Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(f"""
                <div class="thinking-container">
                    {message['thinking']}
                </div>
                """, unsafe_allow_html=True)
        
        # Feedback buttons for assistant messages
        if message['role'] == 'assistant':
            cols = st.columns([1, 1, 6])
            with cols[0]:
                if st.button("👍", key=f"like_{idx}"):
                    st.session_state.feedback[idx] = "liked"
            with cols[1]:
                if st.button("👎", key=f"dislike_{idx}"):
                    st.session_state.feedback[idx] = "disliked"

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    timestamp = datetime.now().strftime('%I:%M %p')
    st.session_state.messages.append({
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
                
                # Store response with metadata
                timestamp = datetime.now().strftime('%I:%M %p')
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": main_response,
                    "thinking": thinking_process,
                    "timestamp": timestamp
                })
                
                # Display response with thinking process
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

# Footer
st.markdown("---")
st.markdown("""
💡 **Tips:**
- Toggle 'Show Thinking Process' in sidebar to see Claude's reasoning
- Use the feedback buttons to mark helpful responses
- Export your chat for later reference
- Adjust temperature for more creative or focused responses
""")
