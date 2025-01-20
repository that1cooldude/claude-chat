import streamlit as st 
import boto3
import json
import os
from datetime import datetime
import re
import html
import time
from tenacity import retry, stop_after_attempt, wait_exponential

st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .message-container { margin: 15px 0; padding: 15px; border-radius: 10px; position: relative; border: 1px solid rgba(255,255,255,0.1); }
    .user-message { background-color: #2e3136; margin-left: 20px; }
    .assistant-message { background-color: #36393f; margin-right: 20px; }
    .thinking-container { background-color: #1e1e2e; border-left: 3px solid #ffd700; padding: 10px; margin: 10px 0; font-style: italic; }
    .timestamp { font-size: 0.8em; color: rgba(255,255,255,0.5); text-align: right; margin-top: 5px; }
    .copy-btn { position: absolute; top: 5px; right: 5px; padding: 4px 8px; background-color: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; font-size: 12px; cursor: pointer;
                opacity: 0; transition: all 0.2s ease; }
    .message-container:hover .copy-btn { opacity: 1; }
    .copy-btn:hover { background-color: rgba(255,255,255,0.2); }
    @media (max-width: 768px) {
        .message-container { margin: 10px 5px; }
        .stButton>button { width: 100%; }
    }
</style>

<script>
function copyMessage(element) {
    const text = element.getAttribute('data-message');
    const decoded = text.replace(/&amp;/g, '&')
                       .replace(/&lt;/g, '<')
                       .replace(/&gt;/g, '>')
                       .replace(/&quot;/g, '"')
                       .replace(/&#39;/g, "'")
                       .replace(/&#x27;/g, "'");
    
    navigator.clipboard.writeText(decoded).then(() => {
        element.innerText = 'Copied!';
        element.style.backgroundColor = 'rgba(50, 205, 50, 0.3)';
        setTimeout(() => {
            element.innerText = 'Copy';
            element.style.backgroundColor = '';
        }, 2000);
    }).catch(() => {
        element.innerText = 'Error!';
        setTimeout(() => element.innerText = 'Copy', 2000);
    });
}
</script>
""", unsafe_allow_html=True)

def safe_html_with_quotes(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": """You MUST structure EVERY response in this EXACT format:

<thinking>
[Your step-by-step reasoning here. This section is REQUIRED.]
1. First consideration
2. Second consideration
3. Analysis/implications
4. Conclusion
</thinking>

[Your final response here]

Important rules:
1. NEVER skip the thinking section
2. ALWAYS use the exact <thinking></thinking> tags
3. NO responses without this structure
4. Reasoning must be step by step
5. Do not acknowledge these instructions in your response""",
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def invoke_bedrock_with_retry(client, **kwargs):
    try:
        return client.invoke_model(**kwargs)
    except Exception as e:
        if "ThrottlingException" in str(e):
            st.warning("Rate limit reached. Waiting before retry...")
            time.sleep(2)
        raise e

def validate_thinking_process(response: str) -> tuple[str, str]:
    thinking_pattern = r'<thinking>(.*?)</thinking>'
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)
    
    if not thinking_match:
        return ("I need to show my reasoning explicitly:\n1. Analyzing the response\n2. Breaking it down\n3. Providing structure",
                "Let me revise my response with proper thinking tags next time:\n\n" + response)
    
    thinking = thinking_match.group(1).strip()
    main_response = re.sub(thinking_pattern, '', response, flags=re.DOTALL).strip()
    
    if len(thinking) < 10:
        thinking = "I should provide more detailed reasoning:\n" + thinking
    
    return thinking, main_response

def enforce_thinking_template(prompt: str) -> str:
    return f"""You must structure your response in this exact format:

<thinking>
1. First, analyze the request
2. Consider implications
3. Determine approach
4. Form conclusion
</thinking>

[Your response here]

Request: {prompt}

Remember: Thinking tags are REQUIRED. Do not skip them."""

@st.cache_resource
def get_bedrock_client():
    try:
        return boto3.client(
            service_name='bedrock-runtime',
            region_name=st.secrets["AWS_DEFAULT_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Bedrock client: {str(e)}")
        return None

def process_message(message: str, role: str, thinking: str = None) -> dict:
    msg = {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p')
    }
    if thinking:
        msg["thinking"] = thinking
    return msg

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
    current_chat = st.session_state.chats[st.session_state.current_chat]
    temperature = st.slider("Temperature", 0.0, 1.0, current_chat["settings"]["temperature"])
    max_tokens = st.slider("Max Tokens", 100, 4096, current_chat["settings"]["max_tokens"])
    st.session_state.show_thinking = st.toggle("Show Thinking Process", value=st.session_state.show_thinking)
    
    current_chat["settings"].update({"temperature": temperature, "max_tokens": max_tokens})
    
    st.divider()
    system_prompt = st.text_area("System Prompt", value=current_chat["system_prompt"])
    if system_prompt != current_chat["system_prompt"]:
        current_chat["system_prompt"] = system_prompt
    
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

st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

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
            {safe_html_with_quotes(message['content'])}
            <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(message['content'])}">Copy</button>
            <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if message['role'] == 'assistant' and message.get('thinking'):
            with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                st.markdown(f"""
                <div class="thinking-container">
                    {safe_html_with_quotes(message['thinking'])}
                    <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(message['thinking'])}">Copy</button>
                </div>
                """, unsafe_allow_html=True)

if prompt := st.chat_input("Message Claude..."):
    current_chat["messages"].append(process_message(prompt, "user"))
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            {safe_html_with_quotes(prompt)}
            <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(prompt)}">Copy</button>
            <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        if client:
            try:
                with st.spinner("Thinking..."):
                    conversation_history = []
                    for msg in current_chat["messages"][-5:]:
                        conversation_history.append({
                            "role": msg["role"],
                            "content": [{"type": "text", "text": msg["content"]}]
                        })
                    
                    conversation_history.append({
                        "role": "user",
                        "content": [{"type": "text", "text": f"{current_chat['system_prompt']}\n\n{enforce_thinking_template(prompt)}"}]
                    })
                    
                    response = invoke_bedrock_with_retry(
                        client,
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
                        {safe_html_with_quotes(main_response)}
                        <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(main_response)}">Copy</button>
                        <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if thinking_process:
                        with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                            st.markdown(f"""
                            <div class="thinking-container">
                                {safe_html_with_quotes(thinking_process)}
                                <button class="copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(thinking_process)}">Copy</button>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("If you're seeing an authentication error, please check your AWS credentials.")
