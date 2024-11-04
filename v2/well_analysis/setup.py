from setuptools import setup, find_packages

setup(
    name="well_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "streamlit>=1.12.0",
        "plotly>=5.3.0",
        "openpyxl>=3.0.9",
        "paramiko>=2.8.1",
        "scipy>=1.7.0",
        "networkx>=2.6.3",
        "pyyaml>=5.4.1",
        "pytest>=6.2.5",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "scikit-learn>=0.24.2"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Well Analysis Tool for Production Optimization",
    python_requires=">=3.8",
)
