#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required Python and Node.js dependencies"""
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"])
    
    print("\nInstalling Node.js dependencies...")
    os.chdir(Path(__file__).parent)
    subprocess.check_call(["npm", "install"])

def build_app():
    """Build the Electron app"""
    print("\nBuilding desktop application...")
    os.chdir(Path(__file__).parent)
    subprocess.check_call(["npm", "run", "build"])

def main():
    try:
        install_dependencies()
        build_app()
        print("\nInstallation complete! You can find the built application in the 'dist' directory.")
    except subprocess.CalledProcessError as e:
        print(f"\nError during installation: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 