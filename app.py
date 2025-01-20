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

# [Previous CSS styles remain the same]

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

# Utility functions remain the same
[Previous utility functions]

# Main interface
st.title("🤖 Enhanced Claude Chat")

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

    # [Previous analytics section remains the same]

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
                # Correct message format
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

                # Correct API call format
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
