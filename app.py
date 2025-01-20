import streamlit as st
import boto3
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Claude Chat",
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
    
    /* Token info */
    .token-info {
        font-size: 0.8em;
        color: #666;
        text-align: right;
        padding-top: 5px;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-2'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

# Calculate cost function
def calculate_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000 * 15) + (output_tokens / 1_000_000 * 75)

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # Model parameters
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 
                          help="Higher values make output more creative but less focused")
    max_tokens = st.slider("Max Tokens", 100, 4096, 1000, 
                          help="Maximum length of the response")
    
    # Stats
    st.divider()
    st.subheader("Session Stats")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# Main chat interface
st.title("🤖 Claude Chat")

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

# Chat input and response
if prompt := st.chat_input("Message Claude..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get Claude's response
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        if client:
            try:
                with st.spinner("Thinking..."):
                    # Make API call to Claude
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
                                    "content": [{"type": "text", "text": prompt}]
                                }
                            ]
                        })
                    )
                    
                    # Parse response
                    response_body = json.loads(response['body'].read())
                    assistant_response = response_body['content'][0]['text']
                    
                    # Calculate usage and cost
                    usage = response_body.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    cost = calculate_cost(input_tokens, output_tokens)
                    
                    # Update session stats
                    st.session_state.total_tokens += (input_tokens + output_tokens)
                    st.session_state.total_cost += cost
                    
                    # Display response
                    st.markdown(assistant_response)
                    st.markdown(
                        f"<div class='token-info'>Tokens: {input_tokens} in / {output_tokens} out | "
                        f"Cost: ${cost:.4f}</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Store message with usage info
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cost": cost
                        }
                    })
                    
            except Exception as e:
                st.error("Failed to get response from Claude")
                st.error(f"Error details: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by Claude 3.5 Sonnet v2 • Built with Streamlit")
