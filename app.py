import streamlit as st
import boto3
import json

# Page config
st.set_page_config(page_title="Claude Chat", layout="wide")

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-2',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💭 Claude Chat")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        ["claude-3-sonnet", "claude-3-haiku"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
                response = get_bedrock_client().invoke_model(
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "messages": [{"role": "user", "content": prompt}]
                    }),
                    modelId=f"anthropic.{model}-20241022-v2:0",
                    accept="application/json",
                    contentType="application/json"
                )
                
                response_body = json.loads(response.get('body').read())
                answer = response_body['content'][0]['text']
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
