import streamlit as st 
import boto3
import json
import os
from datetime import datetime
import re
import html
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO
import csv

st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .message-container { margin: 15px 0; padding: 15px; border-radius: 10px; position: relative; border: 1px solid rgba(255,255,255,0.1); }
    .user-message { background-color: #2e3136; margin-left: 20px; }
    .assistant-message { background-color: #36393f; margin-right: 20px; }
    .thinking-container { background-color: #1e1e2e; border-left: 3px solid #ffd700; padding: 10px; margin: 10px 0; font-style: italic; }
    .timestamp { font-size: 0.8em; color: rgba(255,255,255,0.5); text-align: right; margin-top: 5px; }
    .action-btn { position: absolute; padding: 4px 8px; background-color: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; font-size: 12px; cursor: pointer;
                opacity: 0; transition: all 0.2s ease; }
    .copy-btn { right: 5px; top: 5px; }
    .edit-btn { right: 65px; top: 5px; }
    .delete-btn { right: 115px; top: 5px; }
    .retry-btn { right: 165px; top: 5px; }
    .reaction-btns { position: absolute; right: 215px; top: 5px; opacity: 0; transition: all 0.2s ease; }
    .reaction-btn { padding: 4px 8px; margin: 0 2px; background-color: rgba(255,255,255,0.1);
                   border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; color: #fff; cursor: pointer; }
    .message-container:hover .action-btn, .message-container:hover .reaction-btns { opacity: 1; }
    .action-btn:hover, .reaction-btn:hover { background-color: rgba(255,255,255,0.2); }
    .reaction-btn.active { background-color: rgba(50, 205, 50, 0.3); }
    .favorite-prompt { padding: 10px; margin: 5px 0; background-color: rgba(255,255,255,0.1); 
                      border-radius: 5px; cursor: pointer; }
    .favorite-prompt:hover { background-color: rgba(255,255,255,0.2); }
    .edit-area { margin-top: 10px; background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; }
    @media (max-width: 768px) {
        .message-container { margin: 10px 5px; }
        .action-btn, .reaction-btns { opacity: 1; }
        .reaction-btns { position: relative; right: auto; top: auto; margin-top: 10px; }
        .stButton>button { width: 100%; }
    }
</style>

<script>
function copyMessage(element) {
    const text = element.getAttribute('data-message');
    const parser = new DOMParser();
    const decodedText = parser.parseFromString(`<!doctype html><body>${text}`, 'text/html').body.textContent;
    navigator.clipboard.writeText(decodedText).then(() => {
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

function editMessage(idx) {
    window.streamlitApp.setComponentValue({action: 'edit', messageIdx: idx});
}

function deleteMessage(idx) {
    if (confirm('Delete this message?')) {
        window.streamlitApp.setComponentValue({action: 'delete', messageIdx: idx});
    }
}

function retryMessage(idx) {
    window.streamlitApp.setComponentValue({action: 'retry', messageIdx: idx});
}

function reactToMessage(idx, reaction) {
    window.streamlitApp.setComponentValue({action: 'react', messageIdx: idx, reaction: reaction});
}

function favoritePrompt(prompt) {
    window.streamlitApp.setComponentValue({action: 'favorite', prompt: prompt});
}
</script>
""", unsafe_allow_html=True)

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
if "editing_message" not in st.session_state:
    st.session_state.editing_message = None
if "message_action" not in st.session_state:
    st.session_state.message_action = None
if "favorite_prompts" not in st.session_state:
    st.session_state.favorite_prompts = []
if "reactions" not in st.session_state:
    st.session_state.reactions = {}

def safe_html_with_quotes(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

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
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "reactions": {"likes": 0, "dislikes": 0}
    }
    if thinking:
        msg["thinking"] = thinking
    return msg

def get_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    try:
        with st.spinner("Thinking..."):
            response = invoke_bedrock_with_retry(
                client,
                modelId="arn:aws:bedrock:us-east-2:127214158930:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": settings["max_tokens"],
                    "temperature": settings["temperature"],
                    "messages": conversation_history
                })
            )
            
            response_body = json.loads(response['body'].read())
            full_response = response_body['content'][0]['text']
            
            thinking_process, main_response = validate_thinking_process(full_response)
            return thinking_process, main_response
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None

def export_chat_to_csv(chat):
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Role', 'Content', 'Thinking Process', 'Timestamp', 'Reactions'])
    for message in chat["messages"]:
        writer.writerow([
            message["role"],
            message["content"],
            message.get("thinking", ""),
            message.get("timestamp", ""),
            json.dumps(message.get("reactions", {}))
        ])
    return output.getvalue()

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
    st.subheader("Export Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export JSON"):
            st.download_button(
                "Download JSON",
                data=json.dumps(current_chat, indent=2),
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    with col2:
        if st.button("Export CSV"):
            csv_data = export_chat_to_csv(current_chat)
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"chat_export_{st.session_state.current_chat}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    if st.button("Clear Chat"):
        current_chat["messages"] = []
        st.rerun()
    
    st.divider()
    st.subheader("Favorite Prompts")
    for prompt in st.session_state.favorite_prompts:
        if st.button(f"Use: {prompt[:50]}...", key=f"fav_{prompt}"):
            st.session_state.reuse_prompt = prompt
            st.rerun()

st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")

messages_to_display = current_chat["messages"]
if st.session_state.search_query:
    search_term = st.session_state.search_query.lower()
    messages_to_display = [msg for msg in messages_to_display 
                          if search_term in msg["content"].lower() 
                          or (msg.get("thinking", "").lower() if msg.get("thinking") else "")]

for idx, message in enumerate(messages_to_display):
    with st.chat_message(message["role"]):
        if st.session_state.editing_message == idx and message["role"] == "user":
            edited_message = st.text_area("Edit message", message["content"], key=f"edit_{idx}")
            col1, col2 = st.columns([1,4])
            with col1:
                if st.button("Save", key=f"save_{idx}"):
                    current_chat["messages"][idx]["content"] = edited_message
                    st.session_state.editing_message = None
                    st.rerun()
            with col2:
                if st.button("Cancel", key=f"cancel_{idx}"):
                    st.session_state.editing_message = None
                    st.rerun()
        else:
            action_buttons = ""
            if message["role"] == "user":
                action_buttons = f"""
                <button class="action-btn edit-btn" onclick="editMessage({idx})">Edit</button>
                <button class="action-btn delete-btn" onclick="deleteMessage({idx})">Delete</button>
                """
                if message["content"] not in st.session_state.favorite_prompts:
                    action_buttons += f"""
                    <button class="action-btn favorite-btn" onclick="favoritePrompt('{safe_html_with_quotes(message['content'])}')">
                        Favorite
                    </button>
                    """
            elif message["role"] == "assistant":
                action_buttons = f"""
                <button class="action-btn retry-btn" onclick="retryMessage({idx})">Retry</button>
                <div class="reaction-btns">
                    <button class="reaction-btn like-btn" onclick="reactToMessage({idx}, 'like')">
                        👍 {message.get('reactions', {}).get('likes', 0)}
                    </button>
                    <button class="reaction-btn dislike-btn" onclick="reactToMessage({idx}, 'dislike')">
                        👎 {message.get('reactions', {}).get('dislikes', 0)}
                    </button>
                </div>
                """
            
            st.markdown(f"""
            <div class="message-container {message['role']}-message">
                {safe_html_with_quotes(message['content'])}
                <button class="action-btn copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(message['content'])}">Copy</button>
                {action_buttons}
                <div class="timestamp">{message.get('timestamp', 'No timestamp')}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if message['role'] == 'assistant' and message.get('thinking'):
                with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                    st.markdown(f"""
                    <div class="thinking-container">
                        {safe_html_with_quotes(message['thinking'])}
                        <button class="action-btn copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(message['thinking'])}">Copy</button>
                    </div>
                    """, unsafe_allow_html=True)

if 'message_action' in st.session_state and st.session_state.message_action:
    action = st.session_state.message_action
    if action['action'] == 'edit':
        st.session_state.editing_message = action['messageIdx']
    elif action['action'] == 'delete':
        del current_chat["messages"][action['messageIdx']]
    elif action['action'] == 'retry':
        # Get the last user message before this response
        for i in range(action['messageIdx']-1, -1, -1):
            if current_chat["messages"][i]["role"] == "user":
                prompt = current_chat["messages"][i]["content"]
                current_chat["messages"] = current_chat["messages"][:action['messageIdx']]
                client = get_bedrock_client()
                if client:
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
                    thinking_process, main_response = get_chat_response(
                        prompt,
                        conversation_history,
                        client,
                        current_chat["settings"]
                    )
                    if main_response:
                        current_chat["messages"].append(
                            process_message(main_response, "assistant", thinking_process)
                        )
                break
    elif action['action'] == 'react':
        msg_idx = action['messageIdx']
        reaction = action['reaction']
        if 'reactions' not in current_chat["messages"][msg_idx]:
            current_chat["messages"][msg_idx]['reactions'] = {'likes': 0, 'dislikes': 0}
        current_chat["messages"][msg_idx]['reactions'][f"{reaction}s"] += 1
    elif action['action'] == 'favorite':
        if action['prompt'] not in st.session_state.favorite_prompts:
            st.session_state.favorite_prompts.append(action['prompt'])
    
    st.session_state.message_action = None
    st.rerun()

if prompt := st.chat_input("Message Claude..."):
    current_chat["messages"].append(process_message(prompt, "user"))
    with st.chat_message("user"):
        st.markdown(f"""
        <div class="message-container user-message">
            {safe_html_with_quotes(prompt)}
            <button class="action-btn copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(prompt)}">Copy</button>
            <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with st.chat_message("assistant"):
        client = get_bedrock_client()
        if client:
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
            
            thinking_process, main_response = get_chat_response(
                prompt,
                conversation_history,
                client,
                current_chat["settings"]
            )
            
            if main_response:
                current_chat["messages"].append(
                    process_message(main_response, "assistant", thinking_process)
                )
                
                st.markdown(f"""
                <div class="message-container assistant-message">
                    {safe_html_with_quotes(main_response)}
                    <button class="action-btn copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(main_response)}">Copy</button>
                    <button class="action-btn retry-btn" onclick="retryMessage({len(current_chat['messages'])-1})">Retry</button>
                    <div class="reaction-btns">
                        <button class="reaction-btn like-btn" onclick="reactToMessage({len(current_chat['messages'])-1}, 'like')">👍 0</button>
                        <button class="reaction-btn dislike-btn" onclick="reactToMessage({len(current_chat['messages'])-1}, 'dislike')">👎 0</button>
                    </div>
                    <div class="timestamp">{datetime.now().strftime('%I:%M %p')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if thinking_process:
                    with st.expander("Thinking Process", expanded=st.session_state.show_thinking):
                        st.markdown(f"""
                        <div class="thinking-container">
                            {safe_html_with_quotes(thinking_process)}
                            <button class="action-btn copy-btn" onclick="copyMessage(this)" data-message="{safe_html_with_quotes(thinking_process)}">Copy</button>
                        </div>
                        """, unsafe_allow_html=True)
