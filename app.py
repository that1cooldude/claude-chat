import streamlit as st
import boto3
import json
import os
from datetime import datetime
import re

# Page config
st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Styling
st.markdown("""
<style>
    .message-container { margin: 15px 0; padding: 10px; border-radius: 15px; position: relative; }
    .user-message { background-color: #2e3136; margin-left: 20px; border: 1px solid #404040; }
    .assistant-message { background-color: #36393f; margin-right: 20px; border: 1px solid #404040; }
    .thinking-container { background-color: #1e1e2e; border-left: 3px solid #ffd700; padding: 10px; margin: 10px 0; }
    .timestamp { font-size: 0.8em; color: #666; text-align: right; }
    .copy-btn { position: absolute; top: 5px; right: 5px; padding: 2px 8px; background: #404040; border: none;
                border-radius: 3px; color: white; cursor: pointer; opacity: 0; transition: opacity 0.3s; }
    .message-container:hover .copy-btn { opacity: 1; }
    .search-highlight { background-color: #ffd70066; padding: 0 2px; border-radius: 2px; }
    @media (max-width: 768px) { 
        .message-container { margin: 10px 5px; }
        .stButton>button { width: 100%; }
    }
</style>
<script>
function copyMessage(element) {
    const text = element.getAttribute('data-message');
    navigator.clipboard.writeText(text);
    element.innerText = 'Copied!';
    setTimeout(() => { element.innerText = 'Copy'; }, 2000);
}
</script>
""", unsafe_allow_html=True)

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": """For EVERY response you provide, you must:
1. ALWAYS begin with your reasoning inside <thinking> tags
2. Show your step-by-step thought process
3. Consider multiple aspects of the question
4. Use this EXACT format:

<thinking>
1. [First thought/consideration]
2. [Second thought/consideration]
3. [Additional analysis if needed]
4. [Conclusion or approach]
</thinking>

[Your actual response here]

Never skip the thinking section. Always show your work.""",
            "settings": {"temperature": 0.7, "max_tokens": 1000},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Default"
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

def validate_thinking_process(response: str) -> tuple[str, str]:
    """Validate and extract thinking process and main response."""
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
    if not thinking_match:
        lines = response.split('\n')
        potential_thinking = []
        main_response = []
        in_thinking = False
        for line in lines:
            if line.strip().lower().startswith(('let me think', 'considering', 'analyzing', 'first', '1.', 'step 1')):
                in_thinking = True
            elif line.strip() and not in_thinking:
                main_response.append(line)
            elif in_thinking and line.strip():
                potential_thinking.append(line)
            elif in_thinking and not line.strip() and main_response:
                in_thinking = False
        thinking = '\n'.join(potential_thinking) if potential_thinking else "No explicit thinking process provided"
        main = '\n'.join(main_response) if main_response else response
        return thinking, main
    return thinking_match.group(1).strip(), re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL).strip()

def enforce_thinking_template(prompt: str) -> str:
    """Enhance prompt to enforce thinking process."""
    return f"""Please approach this request carefully, showing ALL your reasoning:
1. Start with <thinking> tags
2. Break down your thought process
3. Consider multiple angles
4. Show your work
5. End thinking tags before response

{prompt}

Remember: Always include <thinking>...</thinking> tags with detailed reasoning."""

@st.cache_resource
def get_bedrock_client():
    """Initialize and return Bedrock client."""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=st.secrets["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
    )

def process_message(message: str, role: str, thinking: str = None) -> dict:
    """Process and format a chat message."""
    msg = {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }
    if thinking:
        msg["thinking"] = thinking
    return msg

# Sidebar
with st.sidebar:
    st.title("Chat Management")
    st.session_state.search_query = st.text_input("🔍 Search messages")
    
    new_chat_name = st.text_input("New Chat Name")
    if st.button("Create Chat") and new_chat_name:
        if re.match(r'^[\w\-\s]+$', new_chat_name):
            if new_chat_name not in st.session_state.chats:
                st.session_state.chats[new_chat_name] = {
                    "messages": [],
                    "system_prompt": st.session_state.chats[st.session_state.current_chat]["system_prompt"],
                    "settings": {"temperature": 0.7, "max_tokens": 1000},
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.current_chat = new_chat_name
                st.rerun()
            else:
                st.error("Chat name already exists")
        else:
            st.error("Invalid chat name. Use only letters, numbers, spaces, and hyphens")
    
    st.session_state.current_chat = st.selectbox("Select Chat", options=list(st.session_state.chats.keys()))
    
    st.divider()
    system_prompt = st.text_area("System Prompt", value=st.session_state.chats[st.session_state.current_chat]["system_prompt"])
    if system_prompt != st.session_state.chats[st.session_state.current_chat]["system_prompt"]:
        st.session_state.chats[st.session_state.current_chat]["system_prompt"] = system_prompt
    
    st.divider()
    current_chat = st.session_state.chats[st.session_state.current_chat]
    temperature = st.slider("Temperature", 0.0, 1.0, current_chat["settings"]["temperature"])
    max_tokens = st.slider("Max Tokens", 100, 4096, current_chat["settings"]["max_tokens"])
    st.session_state.show_thinking = st.toggle("Show Thinking Process", value=st.session_state.show_thinking)
    
    current_chat["settings"].update({"temperature": temperature, "max_tokens": max_tokens})
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Chat"):
            st.download_button(
                "Download JSON",
                data=json.dumps(current_chat, indent=2),
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("Clear Chat"):
            current_chat["messages"] = []
            st.rerun()

# Main chat interface
st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

# Display messages
messages_to_display = current_chat["messages"]
if st.session_state.search_query:
    search_term = st.session_state.search_query.lower()
    messages_to_display = [msg for msg in messages_to_display 
                          if search_term in msg["content"].lower() 
                          or (msg.get("thinking", "").lower() if msg.get("thinking") else "")]

for message in messages_to_display:
    with st.chat_message(message["role"]):
        st.markdown(f"""
        <div class="message-container {message['role']}-message">
            {message['content']}
            <button class="copy-btn" onclick="copyMessage(this)" data-message="{message['content']}">Copy</button>
            <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and 'thinking' in message:
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(f"""
                <div class="thinking-container">
                    {message['thinking']}
                    <button class="copy-btn" onclick="copyMessage(this)" data-message="{message['thinking']}">Copy</button>
                </div>
                """, unsafe_allow_html=True)

# Chat input and response
if prompt := st.chat_input("Message Claude..."):
    current_chat["messages"].append(process_message(prompt, "user"))
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            {prompt}
            <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        try:
            with st.spinner("Thinking..."):
                # Prepare conversation history with thinking enforcement
                enhanced_prompt = f"{enforce_thinking_template(prompt)}"
                
                conversation_history = []
                # Add recent message history for context
                for msg in current_chat["messages"][-5:]:
                    conversation_history.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                
                # Add current prompt with system instructions
                conversation_history.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"{current_chat['system_prompt']}\n\n{enhanced_prompt}"}]
                })
                
                # Make API call
                response = client.invoke_model(
                    modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": current_chat["settings"]["max_tokens"],
                        "temperature": current_chat["settings"]["temperature"],
                        "messages": conversation_history
                    })
                )
                
                response_body = json.loads(response['body'].read())
                full_response = response_body['content'][0]['text']
                
                thinking_process, main_response = validate_thinking_process(full_response)
                
                current_chat["messages"].append(
                    process_message(main_response, "assistant", thinking_process)
                )
                
                st.markdown(f"""
                <div class="message-container assistant-message">
                    {main_response}
                    <button class="copy-btn" onclick="copyMessage(this)" data-message="{main_response}">Copy</button>
                    <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if thinking_process:
                    with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {thinking_process}
                            <button class="copy-btn" onclick="copyMessage(this)" data-message="{thinking_process}">Copy</button>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("If you're seeing an authentication error, please check your AWS credentials.")
