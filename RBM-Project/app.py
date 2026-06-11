import streamlit as st
import os

# Set page characteristics to wide layout
st.set_page_config(layout="wide", page_title="Telco Connect RCS RBM Prototype")

# Find the path to your HTML file relative to this script
current_dir = os.path.dirname(__file__)
html_file_path = os.path.join(current_dir, "index.html")

# Read the raw HTML code
with open(html_file_path, "r", encoding="utf-8") as f:
    html_source = f.read()

# Render the HTML directly into the page app container (Bypasses iframe sandboxing)
st.markdown(html_source, unsafe_allow_html=True)
