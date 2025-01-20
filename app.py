import streamlit as st
import boto3
import json
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Claude 3 Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# Styling
st.markdown("""
<style>
    .stChat {
        padding: 20px;
    }
    .user-message {
        background-color: #2e3136;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #36393f;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .token-info {
        font-size: 0.8em;
        color: #666;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_bedrock_client():
    """Initialize Bedrock client with error handling"""
    try:
        if not st.secrets.get("AWS_ACCESS_KEY_ID") or not st.secrets.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS credentials not found in secrets")
        
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-2',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

def invoke_claude(client, prompt, max_tokens=1000, temperature=0.7):
    """Invoke Claude model with exact catalog format"""
    try:
        response = client.invoke_model(
            modelId="anthropic.claude-3",  # Changed to base model ID
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "top_k": 250,
                "stop_sequences": [],
                "temperature": temperature,
                "top_p": 0.999,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            })
        )
        
        return json.loads(response.get('body').read())
    except Exception as e:
        st.error(f"Error invoking Claude: {str(e)}")
        if "ValidationException" in str(e):
            st.info("""
            Before using this app:
            1. Go to AWS Console -> Bedrock
            2. Navigate to Model access
            3. Enable Claude 3 model access
            4. Create an inference profile for Claude 3
            5. Note: You may need to request model access from Anthropic first
            """)
        raise e


def calculate_cost(input_tokens, output_tokens):
    """Calculate cost based on token usage"""
    return (input_tokens / 1_000_000 * 15) + (output_tokens / 1_000_000 * 75)

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4096, 1000)
    
    st.divider()
    
    st.subheader("Session Stats")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# Main chat interface
st.title("🤖 Claude 3 Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "usage" in message:
            st.markdown(
                f"<div class='token-info'>Tokens: {message['usage']['input_tokens']} in / "
                f"{message['usage']['output_tokens']} out | "
                f"Cost: ${message['usage']['cost']:.4f}</div>",
                unsafe_allow_html=True
            )

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Initialize client
    client = get_bedrock_client()
    if not client:
        st.error("Failed to initialize Bedrock client. Please check your AWS credentials.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get Claude's response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = invoke_claude(
                    client=client,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            if response and 'content' in response:
                answer = response['content'][0]['text']
                
                # Calculate usage and cost
                usage = response.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                cost = calculate_cost(input_tokens, output_tokens)
                
                # Update session stats
                st.session_state.total_tokens += (input_tokens + output_tokens)
                st.session_state.total_cost += cost
                
                # Display response
                st.markdown(answer)
                st.markdown(
                    f"<div class='token-info'>Tokens: {input_tokens} in / {output_tokens} out | "
                    f"Cost: ${cost:.4f}</div>",
                    unsafe_allow_html=True
                )
                
                # Store message with usage info
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost": cost
                    }
                })
                
        except Exception as e:
            st.error("Failed to get response from Claude")
            if "ValidationException" in str(e):
                st.info("""
                Troubleshooting steps:
                1. Check AWS Console -> Bedrock -> Model access
                2. Verify model ID matches exactly
                3. Ensure region is correct (us-east-2)
                4. Confirm model access is enabled
                """)
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Powered by Claude 3 Sonnet on AWS Bedrock</div>",
    unsafe_allow_html=True
)
