"""
PRZZ Mollifier Explorer - Streamlit Cloud Entry Point

This file is the entry point for Streamlit Cloud deployment.
Streamlit Cloud looks for app.py or streamlit_app.py in the repository root.

Usage:
    streamlit run streamlit_app.py

Or via Streamlit Cloud:
    1. Connect repository to share.streamlit.io
    2. Set main file path to: streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main app
from streamlit_app.app import main

if __name__ == "__main__":
    main()
