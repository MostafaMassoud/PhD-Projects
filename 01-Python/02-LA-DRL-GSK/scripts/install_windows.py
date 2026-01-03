"""
Windows Installation Script for LA-DRL-GSK
==========================================

Run with:
    python scripts/install_windows.py

This script:
1. Installs PyTorch CPU from the special index URL
2. Installs all other dependencies
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
    print("  LA-DRL-GSK Windows Installation")
    print("="*60)
    
    # Step 1: Upgrade pip
    if not run_cmd(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    ):
        return 1
    
    # Step 2: Install PyTorch CPU
    if not run_cmd(
        f"{sys.executable} -m pip install torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu",
        "Installing PyTorch (CPU with MKL)"
    ):
        return 1
    
    # Step 3: Install other dependencies
    if not run_cmd(
        f"{sys.executable} -m pip install -r requirements-windows.txt",
        "Installing other dependencies"
    ):
        return 1
    
    # Step 4: Verify installation
    print("\n" + "="*60)
    print("  Verifying Installation")
    print("="*60 + "\n")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  MKL available: {torch.backends.mkl.is_available()}")
        print(f"  Threads: {torch.get_num_threads()}")
        
        import numpy as np
        print(f"  NumPy version: {np.__version__}")
        
        print("\n✅ Installation complete!")
        print("\nRecommended thread configuration:")
        print("  torch.set_num_threads(16)  # = physical cores")
        print("  torch.set_num_interop_threads(2)")
        
    except ImportError as e:
        print(f"  ❌ Verification failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
