"""
SecuroServ - AI-Powered Security Surveillance System
Entry point for the application.
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import SecuroServApp

if __name__ == "__main__":
    app = SecuroServApp()
    app.run()
