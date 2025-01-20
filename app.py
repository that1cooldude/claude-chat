import streamlit as st
import boto3
import json
from datetime import datetime
import plotly.express as px
import pandas as pd
import re
import time

# Page configuration
st.set_page_config(
    page_title="Enhanced Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #2e3a48;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    
    .assistant-message {
        background-color: #1a1f2f;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e1e1e !important;
        border-radius: 5px;
    }
    
    /* Thinking tags */
    .thinking-tag {
        background-color: #3d2f00;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    
    /* Token counter */
    .token-counter {
        font-size: 0.8rem;
        color: #666;
        text-align: right;
        margin-top: 5px;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        .user-message, .assistant-message {
            padding: 0.8rem;
            margin: 0.8rem 0;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'start_time' not in st.session_state:
    st.session_state.start_time = datetime.now()
if 'token_history' not in st.session_state:
    st.session_state.token_history = []

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-2',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

# Utility functions
def calculate_cost(input_tokens, output_tokens, model):
    if "sonnet" in model.lower():
        return (input_tokens / 1_000_000 * 15) + (output_tokens / 1_000_000 * 75)
    else:  # haiku
        return (input_tokens / 1_000_000 * 5) + (output_tokens / 1_000_000 * 25)

def format_message(content):
    # Format code blocks
    content = re.sub(
        r'```([\w]*)\n(.*?)```',
        r'<div class="code-block"><div class="code-header">\1</div><pre><code>\2</code></pre></div>',
        content,
        flags=re.DOTALL
    )
    
    # Format thinking tags
    content = re.sub(
        r'<thinking>(.*?)</thinking>',
        r'<div class="thinking-tag">🤔 \1</div>',
        content,
        flags=re.DOTALL
    )
    
    return content

# Main interface
st.title("🤖 Enhanced Claude Chat")

# Debug expander in sidebar
with st.sidebar:
    with st.expander("Debug Info"):
        st.write({
            "AWS Region": "us-east-2",
            "Bedrock Access": "Checking...",
            "Available Models": "..."
        })

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model selection with CORRECT IDs
    model = st.selectbox(
        "Model",
        [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        ],
        format_func=lambda x: "Claude 3.5 Sonnet" if "sonnet" in x else "Claude 3.5 Haiku"
    )
    
    # Parameters
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1500)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant. Always show your reasoning using <thinking> tags.",
        height=100
    )
    
    # Analytics
    st.header("Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    with col2:
        st.metric("Total Cost", f"${st.session_state.total_cost:.2f}")
    
    # Token usage graph
    if st.session_state.token_history:
        df = pd.DataFrame(st.session_state.token_history)
        fig = px.line(
            df,
            x='timestamp',
            y=['input_tokens', 'output_tokens'],
            title='Token Usage'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export chat
    if st.button("Export Chat"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_data = {
            "messages": st.session_state.messages,
            "analytics": {
                "total_tokens": st.session_state.total_tokens,
                "total_cost": st.session_state.total_cost,
                "session_start": st.session_state.start_time.isoformat(),
                "token_history": st.session_state.token_history
            }
        }
        st.download_button(
            "Download Chat History",
            data=json.dumps(export_data, indent=2),
            file_name=f"chat_export_{timestamp}.json",
            mime="application/json"
        )
    
    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
chat_container = st.container()

with chat_container:
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(format_message(msg["content"]), unsafe_allow_html=True)
            if "usage" in msg:
                st.markdown(
                    f'<div class="token-counter">Tokens: {msg["usage"]["input_tokens"]} in / {msg["usage"]["output_tokens"]} out</div>',
                    unsafe_allow_html=True
                )

# Chat input
if prompt := st.chat_input("Your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get Claude's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare messages with correct format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{system_prompt}\n\n{prompt}"
                            }
                        ]
                    }
                ]

                # Make API call with correct format
                response = get_bedrock_client().invoke_model(
                    modelId=model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_k": 250,
                        "top_p": 0.999,
                        "stopSequences": [],
                        "messages": messages
                    })
                )
                
                response_body = json.loads(response.get('body').read())
                answer = response_body['content'][0]['text']
                usage = response_body.get('usage', {})
                
                # Update analytics
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                cost = calculate_cost(input_tokens, output_tokens, model)
                
                st.session_state.total_tokens += input_tokens + output_tokens
                st.session_state.total_cost += cost
                
                st.session_state.token_history.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost
                })
                
                # Display response
                st.markdown(format_message(answer), unsafe_allow_html=True)
                st.markdown(
                    f'<div class="token-counter">Tokens: {input_tokens} in / {output_tokens} out | Cost: ${cost:.3f}</div>',
                    unsafe_allow_html=True
                )
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "usage": usage
                })
                
            except Exception as e:
                st.error(f"Error Details: {str(e)}")
                if "ValidationException" in str(e):
                    st.info("""
                    Troubleshooting steps:
                    1. Check AWS Console -> Bedrock -> Model access
                    2. Verify model ID matches exactly
                    3. Ensure region is correct (us-east-2)
                    4. Confirm model access is enabled
                    """)

# Session info
with st.expander("Session Information"):
    st.write(f"Session started: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.write(f"Messages: {len(st.session_state.messages)}")
    st.write(f"Total tokens: {st.session_state.total_tokens:,}")
    st.write(f"Total cost: ${st.session_state.total_cost:.2f}")
