"""
PHENIX Interactive Research Dashboard - Main Application Entrypoint
Compatible with local execution, Streamlit CLI, and Replit deployment.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.visualization.dashboard import render_dashboard, run_dashboard_cli

if __name__ == "__main__":
    try:
        import streamlit as st
        # If running inside 'streamlit run app.py'
        if "streamlit" in sys.modules:
            render_dashboard()
        else:
            run_dashboard_cli()
    except ImportError:
        run_dashboard_cli()
