import streamlit as st
import boto3
import json
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import csv
from io import StringIO

# Page Configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Message styling */
    .message-container {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        position: relative;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .user-message {
        background-color: rgba(70, 70, 70, 0.2);
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: rgba(50, 50, 50, 0.2);
        margin-right: 2rem;
    }
    
    .message-content {
        white-space: pre-wrap;
        word-break: break-word;
        margin-right: 100px;
        font-family: sans-serif;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Button styling */
    .message-actions {
        position: absolute;
        right: 1rem;
        top: 1rem;
        display: flex;
        gap: 0.5rem;
    }
    
    .action-button {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        cursor: pointer;
        font-size: 0.8rem;
        transition: all 0.2s;
    }
    
    .action-button:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }
    
    .action-button.copied {
        background-color: #4CAF50;
    }
    
    /* Timestamp */
    .timestamp {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        text-align: right;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Thinking Process */
    .thinking-container {
        background-color: rgba(30, 30, 46, 0.5);
        border-left: 3px solid #ffd700;
        padding: 1rem;
        margin-top: 0.5rem;
        font-style: italic;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* Hide HTML tags */
    .stMarkdown div {
        overflow: hidden;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .message-actions {
            position: relative;
            right: auto;
            top: auto;
            opacity: 1;
            margin-top: 0.5rem;
        }
        .message-content {
            margin-right: 0;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {display: none;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 1000
if 'show_thinking' not in st.session_state:
    st.session_state.show_thinking = True

# AWS Bedrock client setup
@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-2'
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ai_response(prompt: str, temperature: float, max_tokens: int) -> tuple[str, str]:
    client = get_bedrock_client()
    if not client:
        return None, "Failed to initialize AI client"
    
    try:
        # Add the thinking structure requirement to the prompt
        structured_prompt = f"\n\nHuman: You MUST structure your response with <contemplator> for your thinking process and <final_answer> for your main response.\n\n{prompt}\n\nAssistant:"
        
        response = client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps({
                "prompt": structured_prompt,
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        
        response_body = json.loads(response['body'].read())
        full_response = response_body['completion']
        
        # Extract thinking and response using the correct tags
        thinking = ""
        main_response = full_response
        
        contemplator_start = full_response.find('<contemplator>')
        contemplator_end = full_response.find('</contemplator>')
        final_answer_start = full_response.find('<final_answer>')
        final_answer_end = full_response.find('</final_answer>')
        
        if contemplator_start != -1 and contemplator_end != -1:
            thinking = full_response[contemplator_start + 13:contemplator_end].strip()
            
        if final_answer_start != -1 and final_answer_end != -1:
            main_response = full_response[final_answer_start + 13:final_answer_end].strip()
        else:
            main_response = full_response.strip()
        
        return thinking, main_response
        
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return None, str(e)

def add_message(role: str, content: str, thinking: str = None):
    """Add a message to the conversation history"""
    st.session_state.messages.append({
        'role': role,
        'content': content.strip(),
        'thinking': thinking.strip() if thinking else None,
        'timestamp': datetime.now().strftime('%I:%M %p')
    })

def export_chat():
    """Export chat history to CSV"""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking', 'Timestamp'])
    
    for msg in st.session_state.messages:
        writer.writerow([
            msg['role'],
            msg['content'],
            msg.get('thinking', ''),
            msg['timestamp']
        ])
    
    return output.getvalue()

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    st.session_state.temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1
    )
    
    st.session_state.max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=st.session_state.max_tokens,
        step=100
    )
    
    st.session_state.show_thinking = st.toggle(
        "Show Thinking Process",
        value=st.session_state.show_thinking
    )
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("Export Chat"):
        csv_data = export_chat()
        st.download_button(
            label="Download Chat",
            data=csv_data,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# Main chat interface
st.title("💭 AI Chat Assistant")

# Display messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Add action buttons
        col1, col2, col3, col4 = st.columns([1,1,1,6])
        with col1:
            if st.button('Copy', key=f'copy_{idx}'):
                st.write('Copied!')
        
        if message['role'] == 'user':
            with col2:
                if st.button('Edit', key=f'edit_{idx}'):
                    st.session_state['editing'] = idx
            with col3:
                if st.button('Delete', key=f'delete_{idx}'):
                    st.session_state.messages.pop(idx)
                    st.rerun()
        
        elif message['role'] == 'assistant':
            with col2:
                if st.button('Retry', key=f'retry_{idx}'):
                    st.session_state['retrying'] = idx
                    st.rerun()
        
        # Display thinking process for assistant messages
        if message['role'] == 'assistant' and message.get('thinking'):
            with st.expander("💭 Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(message['thinking'])

# Chat input
if prompt := st.chat_input("Message the AI..."):
    add_message('user', prompt)
    
    with st.spinner("AI is thinking..."):
        thinking, response = get_ai_response(
            prompt,
            st.session_state.temperature,
            st.session_state.max_tokens
        )
        
        if response:
            add_message('assistant', response, thinking)
            st.rerun()
        else:
            st.error("Failed to get AI response. Please try again.")

# Handle editing
if 'editing' in st.session_state:
    idx = st.session_state['editing']
    new_content = st.text_input("Edit message", st.session_state.messages[idx]["content"])
    if st.button("Save"):
        st.session_state.messages[idx]["content"] = new_content
        del st.session_state['editing']
        st.rerun()

# Handle retrying
if 'retrying' in st.session_state:
    idx = st.session_state['retrying']
    # Find the last user message before this one
    for i in range(idx - 1, -1, -1):
        if st.session_state.messages[i]["role"] == "user":
            prompt = st.session_state.messages[i]["content"]
            # Remove messages after the user message
            st.session_state.messages = st.session_state.messages[:idx]
            # Get new AI response
            thinking, response = get_ai_response(
                prompt,
                st.session_state.temperature,
                st.session_state.max_tokens
            )
            if response:
                add_message('assistant', response, thinking)
            break
    del st.session_state['retrying']
    st.rerun()
