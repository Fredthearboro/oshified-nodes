"""
Oshified ComfyUI Nodes
Auto-installs requirements on first import
"""

import os
import subprocess
import sys

def install_requirements():
    """Install requirements if not already installed"""
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    # Check if all required packages are installed
    missing_packages = []
    
    try:
        import yt_dlp
    except ImportError:
        missing_packages.append("yt-dlp")
    
    try:
        import boto3
    except ImportError:
        missing_packages.append("boto3")
    
    if not missing_packages:
        print("[Oshified] ✅ All requirements already installed")
        return True
    
    # Install missing packages
    print(f"[Oshified] Installing missing packages: {', '.join(missing_packages)}")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", requirements_file, 
            "--no-warn-script-location",
            "-q"  # Quiet mode
        ])
        print("[Oshified] ✅ Requirements installed successfully")
        return True
    except Exception as e:
        print(f"[Oshified] ❌ Failed to install requirements: {e}")
        print(f"[Oshified] Please manually run: pip install -r {requirements_file}")
        return False

# Auto-install on import
install_requirements()

# Import node definitions
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
