#!/usr/bin/env python3
"""
SecureLLM - AI Red Teaming Platform by HCLTech
===============================================

A lightweight, production-ready web application for comprehensive red teaming
of Large Language Models. Powered by Garak vulnerability scanner.

Tagline: Red Team Your AI Before Adversaries Do

Developed by: HCLTech
Author: Isi Idemudia
Version: 2.0.0
License: Apache 2.0
"""

# This is the main entry point for SecureLLM
# Import and run the Streamlit application

import os
import sys

# Set application metadata
os.environ['SECURELLM_VERSION'] = '2.0.0'
os.environ['SECURELLM_VENDOR'] = 'HCLTech'

# Import the main application
from garak_streamlit_app import main

if __name__ == "__main__":
    print("=" * 60)
    print("  SecureLLM - AI Red Teaming Platform")
    print("  Red Team Your AI Before Adversaries Do")
    print("  Developed by HCLTech")
    print("  Version 2.0.0")
    print("=" * 60)
    print("\n🚀 Starting SecureLLM...\n")

    main()
