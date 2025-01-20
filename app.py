import streamlit as st 
import boto3
import json
import os
from datetime import datetime
import re
import html
import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO
import csv
import wolframalpha
import yfinance as yf
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "Default": {
            "messages": [],
            "system_prompt": """You MUST structure EVERY response with thinking and final answer sections. You have access to several tools:
            1. Calculator - For mathematical computations
            2. Stock Data - For financial market information
            3. Weather - For current weather data
            4. Web Search - For recent information
            5. Image Analysis - For analyzing images
            When appropriate, use these tools to enhance your responses.""",
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
if "tools_enabled" not in st.session_state:
    st.session_state.tools_enabled = {
        "calculator": True,
        "stocks": True,
        "weather": True,
        "web_search": True,
        "image_analysis": True
    }
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "wolfram_alpha": st.secrets.get("WOLFRAM_ALPHA_KEY", ""),
        "weather_api": st.secrets.get("WEATHER_API_KEY", ""),
        "serp_api": st.secrets.get("SERP_API_KEY", "")
    }

# Add the previous CSS styles here (unchanged)
# [Previous CSS code remains the same]

# Tool Functions
def calculate(query):
    """Use Wolfram Alpha for calculations"""
    try:
        client = wolframalpha.Client(st.session_state.api_keys["wolfram_alpha"])
        res = client.query(query)
        return next(res.results).text
    except Exception as e:
        return f"Error in calculation: {str(e)}"

def get_stock_data(symbol, period="1mo"):
    """Get stock market data"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        # Create an interactive plot
        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
        
        fig.update_layout(title=f"{symbol} Stock Price",
                         yaxis_title='Price (USD)',
                         xaxis_title='Date')
        
        return {
            "current_price": stock.info['regularMarketPrice'],
            "change": stock.info['regularMarketChangePercent'],
            "volume": stock.info['regularMarketVolume'],
            "plot": fig
        }
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

def get_weather(city):
    """Get weather data for a city"""
    try:
        api_key = st.session_state.api_keys["weather_api"]
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
        else:
            return f"Error: {data['message']}"
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

def web_search(query):
    """Perform a web search"""
    try:
        api_key = st.session_state.api_keys["serp_api"]
        url = f"https://serpapi.com/search?q={query}&api_key={api_key}"
        response = requests.get(url)
        data = response.json()
        return data.get("organic_results", [])[:3]
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def analyze_image(image):
    """Analyze an uploaded image"""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Use AWS Rekognition for image analysis
        client = boto3.client('rekognition',
                            region_name=st.secrets["AWS_DEFAULT_REGION"],
                            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
        
        response = client.detect_labels(Image={'Bytes': img_byte_arr})
        return response['Labels']
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Enhanced Message Processing
def process_message_with_tools(message: str, role: str, thinking: str = None, tools_output: dict = None) -> dict:
    """Process message with additional tool outputs"""
    return {
        "role": role,
        "content": message,
        "timestamp": datetime.now().strftime('%I:%M %p'),
        "reactions": {"likes": 0, "dislikes": 0},
        "thinking": thinking,
        "tools_output": tools_output
    }

# Enhanced Chat Response
def get_enhanced_chat_response(prompt: str, conversation_history: list, client, settings: dict):
    """Get response from Claude with tool integration"""
    try:
        # Check if prompt requires tool use
        tools_output = {}
        
        # Calculator
        if st.session_state.tools_enabled["calculator"] and re.search(r'calculate|compute|solve|math', prompt.lower()):
            tools_output["calculator"] = calculate(prompt)
        
        # Stocks
        if st.session_state.tools_enabled["stocks"] and re.search(r'stock|price|market|ticker', prompt.lower()):
            stock_symbol = re.search(r'([A-Z]{1,5})', prompt.upper())
            if stock_symbol:
                tools_output["stocks"] = get_stock_data(stock_symbol.group(1))
        
        # Weather
        if st.session_state.tools_enabled["weather"] and re.search(r'weather|temperature|forecast', prompt.lower()):
            city_match = re.search(r'in ([a-zA-Z\s]+)', prompt)
            if city_match:
                tools_output["weather"] = get_weather(city_match.group(1))
        
        # Web Search
        if st.session_state.tools_enabled["web_search"] and re.search(r'search|find|look up|recent', prompt.lower()):
            tools_output["web_search"] = web_search(prompt)
        
        # Enhance prompt with tool outputs
        enhanced_prompt = prompt
        if tools_output:
            enhanced_prompt += "\n\nTool outputs:\n" + json.dumps(tools_output, indent=2)
        
        # Get Claude's response
        response = invoke_bedrock_with_retry(
            client,
            modelId="anthropic.claude-v2",
            body=json.dumps({
                "prompt": f"\n\nHuman: {enhanced_prompt}\n\nAssistant:",
                "max_tokens_to_sample": settings["max_tokens"],
                "temperature": settings["temperature"],
                "anthropic_version": "bedrock-2023-05-31"
            })
        )
        
        response_body = json.loads(response['body'].read())
        full_response = response_body['completion']
        
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            main_response = re.sub(r'<thinking>.*?</thinking>', '', full_response, flags=re.DOTALL).strip()
        else:
            thinking = "Reasoning process not explicitly provided"
            main_response = full_response
        
        return thinking, main_response, tools_output
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None

# Enhanced Message Display
def display_enhanced_message(message: dict, idx: int, current_chat: dict):
    """Display message with tool outputs"""
    # Display regular message content
    display_message(message, idx, current_chat)
    
    # Display tool outputs if present
    if message.get("tools_output"):
        with st.expander("Tool Outputs", expanded=True):
            for tool, output in message["tools_output"].items():
                st.subheader(tool.title())
                if tool == "stocks" and isinstance(output, dict) and "plot" in output:
                    # Display stock plot
                    st.plotly_chart(output["plot"])
                    st.write(f"Current Price: ${output['current_price']:.2f}")
                    st.write(f"Change: {output['change']:.2f}%")
                elif tool == "weather" and isinstance(output, dict):
                    # Display weather info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Temperature", f"{output['temperature']}°C")
                        st.metric("Humidity", f"{output['humidity']}%")
                    with col2:
                        st.metric("Wind Speed", f"{output['wind_speed']} m/s")
                        st.write(f"Description: {output['description']}")
                elif tool == "web_search" and isinstance(output, list):
                    # Display search results
                    for result in output:
                        st.write(f"🔍 [{result['title']}]({result['link']})")
                        st.write(result['snippet'])
                else:
                    st.write(output)

# Enhanced Sidebar
def render_enhanced_sidebar():
    with st.sidebar:
        st.title("Chat Settings")
        
        # Tools Section
        st.subheader("Tools")
        for tool in st.session_state.tools_enabled:
            st.session_state.tools_enabled[tool] = st.toggle(
                f"Enable {tool.replace('_', ' ').title()}",
                value=st.session_state.tools_enabled[tool]
            )
        
        # API Keys Section
        st.subheader("API Keys")
        for key in st.session_state.api_keys:
            st.session_state.api_keys[key] = st.text_input(
                f"{key.replace('_', ' ').title()} API Key",
                value=st.session_state.api_keys[key],
                type="password"
            )
        
        # Previous sidebar content
        # [Previous sidebar code remains the same]

# Main Application Logic
def main():
    render_enhanced_sidebar()
    
    st.title(f"🤖 Claude Chat - {st.session_state.current_chat}")
    
    # Image upload for analysis
    if st.session_state.tools_enabled["image_analysis"]:
        uploaded_file = st.file_uploader("Upload an image for analysis", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    analysis_results = analyze_image(image)
                    st.write("Analysis Results:")
                    for label in analysis_results:
                        st.write(f"- {label['Name']}: {label['Confidence']:.2f}%")
    
    # Search functionality
    st.session_state.search_query = st.text_input("🔍 Search messages")
    
    # Display messages
    current_chat = st.session_state.chats[st.session_state.current_chat]
    messages_to_display = current_chat["messages"]
    
    if st.session_state.search_query:
        search_term = st.session_state.search_query.lower()
        messages_to_display = [
            msg for msg in messages_to_display 
            if search_term in msg["content"].lower() 
            or search_term in (msg.get("thinking", "").lower())
        ]
    
    for idx, message in enumerate(messages_to_display):
        with st.chat_message(message["role"]):
            display_enhanced_message(message, idx, current_chat)
    
    # Chat input
    if prompt := st.chat_input("Message Claude..."):
        # Add user message
        current_chat["messages"].append(process_message_with_tools(prompt, "user"))
        
        # Get and display assistant response
        client = get_bedrock_client()
        if client:
            thinking_process, main_response, tools_output = get_enhanced_chat_response(
                prompt,
                current_chat["messages"][-5:],
                client,
                current_chat["settings"]
            )
            
            if main_response:
                current_chat["messages"].append(
                    process_message_with_tools(
                        main_response,
                        "assistant",
                        thinking_process,
                        tools_output
                    )
                )
                st.rerun()

if __name__ == "__main__":
    main()