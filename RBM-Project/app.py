import streamlit as st
import streamlit.components.v1 as components
import os

# Set page characteristics to wide/responsive layout
st.set_page_config(layout="wide", page_title="Verizon RCS RBM Prototype")

# Find the path to your HTML file relative to this script
current_dir = os.path.dirname(__file__)
html_file_path = os.path.join(current_dir, "index.html")

# Read the raw HTML code
with open(html_file_path, "r", encoding="utf-8") as f:
    html_source = f.read()

# Render the HTML in Streamlit using an iframe component
components.html(html_source, height=850, scrolling=True)
