"""
Centralized styles configuration for Claude Chat
"""

CUSTOM_CSS = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1rem;
}
.user-bubble {
    background-color: #2e3136; /* dark gray for user */
    color: #fff;
    padding: 12px 16px;
    border-radius: 10px;
    max-width: 80%;
    align-self: flex-end;
    word-wrap: break-word;
    margin-bottom: 4px;
}
.assistant-bubble {
    background-color: #454a50; /* lighter gray for assistant */
    color: #fff;
    padding: 12px 16px;
    border-radius: 10px;
    max-width: 80%;
    align-self: flex-start;
    word-wrap: break-word;
    margin-bottom: 4px;
}
.timestamp {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
    margin-top: 2px;
    text-align: right;
}
.thinking-expander {
    background-color: #333;
    border-left: 3px solid #ffd700;
    border-radius: 5px;
    padding: 8px;
    margin-top: 4px;
}
.thinking-text {
    color: #ffd700;
    font-style: italic;
    white-space: pre-wrap;
    word-break: break-word;
}
</style>
"""
