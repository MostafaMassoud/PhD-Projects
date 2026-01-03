"""
LA-DRL-GSK: Landscape-Aware Deep Reinforcement Learning GSK Algorithm
=====================================================================

Installation:
    pip install -e .
    
Or with visualization support:
    pip install -e ".[viz]"
"""
from setuptools import setup, find_packages

setup(
    name="la_drl_gsk",
    version="1.0.0",
    author="LA-DRL-GSK Research Team",
    description="Landscape-Aware Deep Reinforcement Learning GSK Algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/LA-DRL-GSK",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        # PyTorch is optional - works without it using adaptive heuristic
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],  # For neural network policy
        "dev": ["pytest", "pytest-cov", "black", "flake8", "mypy"],
        "viz": ["matplotlib>=3.4.0", "seaborn>=0.11.0"],
        "all": ["torch>=2.0.0", "matplotlib>=3.4.0", "seaborn>=0.11.0", "pytest"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "la-drl-gsk=run:main",
        ],
    },
)
