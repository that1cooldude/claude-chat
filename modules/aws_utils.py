"""
AWS utility functions for Claude Chat
"""

import boto3
import streamlit as st
from botocore.exceptions import ClientError

@st.cache_resource
def get_bedrock_client():
    """Get Bedrock client with error handling"""
    try:
        return boto3.client(
            "bedrock-runtime",
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Error creating Bedrock client: {e}")
        return None

@st.cache_resource
def get_s3_client():
    """Get S3 client with error handling"""
    try:
        return boto3.client(
            "s3",
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-2"),
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
    except Exception as e:
        st.error(f"Error creating S3 client: {e}")
        return None
