import streamlit as st
import boto3
import json
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Claude Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=st.secrets["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

# Main chat interface
st.title("🤖 Claude Chat")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Message Claude..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get Claude's response
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        try:
            with st.spinner("Thinking..."):
                response = client.invoke_model(
                    modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}]
                            }
                        ]
                    })
                )
                
                response_body = json.loads(response['body'].read())
                assistant_response = response_body['content'][0]['text']
                
                # Store and display response
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                st.markdown(assistant_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Sidebar controls
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
