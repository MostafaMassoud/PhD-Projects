"""
macOS Apple Silicon Installation Script for LA-DRL-GSK
======================================================

Run with:
    python3 scripts/install_mac.py

This script installs all dependencies for Apple Silicon Macs (M1/M2/M3).
MPS (Metal Performance Shaders) is automatically available.
"""

import subprocess
import sys


def run_cmd(cmd, desc):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"  > {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {desc}")
        return False
    print(f"\n✅ Success: {desc}")
    return True


def main():
    print("\n" + "="*60)
    print("  LA-DRL-GSK macOS Apple Silicon Installation")
    print("="*60)
    
    # Step 1: Upgrade pip
    if not run_cmd(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    ):
        return 1
    
    # Step 2: Install all dependencies
    if not run_cmd(
        f"{sys.executable} -m pip install -r requirements-mac-m3.txt",
        "Installing dependencies"
    ):
        return 1
    
    # Step 3: Verify installation
    print("\n" + "="*60)
    print("  Verifying Installation")
    print("="*60 + "\n")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")
        
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        
        print("\n✅ Installation complete!")
        print("\nMPS (Metal) is available for GPU acceleration.")
        print("To use MPS in your code:")
        print("  device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')")
        
    except ImportError as e:
        print(f"  ❌ Verification failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
