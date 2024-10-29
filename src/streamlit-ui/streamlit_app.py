import requests
import streamlit as st

# OpenSearch configuration
opensearch_url = "http://opensearch:9200"

# Label Studio configuration
label_studio_url = "http://label-studio:8080"

st.title("Streamlit Interface for OpenSearch and Label Studio")

# Test OpenSearch Connection
try:
    response = requests.get(f"{opensearch_url}/_cluster/health")
    st.write("OpenSearch Status:", response.json())
except requests.ConnectionError:
    st.write("Error: Unable to connect to OpenSearch.")

# Test Label Studio Connection
try:
    response = requests.get(f"{label_studio_url}/api/projects", headers={"Authorization": "Token YOUR_LABEL_STUDIO_API_KEY"})
    st.write("Label Studio Projects:", response.json())
except requests.ConnectionError:
    st.write("Error: Unable to connect to Label Studio.")
