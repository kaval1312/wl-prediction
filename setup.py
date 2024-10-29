from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="oil-well-calculator",
    version="1.0.0",
    author="Test WL",
    author_email="akovalenko1312@gmail.com",
    description="A comprehensive oil well abandonment analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kaval1312/oil-well-calculator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.24.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "plotly>=5.15.0",
        "scipy>=1.10.1",
        "statsmodels>=0.14.0",
        "pyyaml>=6.0.1",
        "pytest>=7.4.0",
    ],
    entry_points={
        "console_scripts": [
            "oil-well-calculator=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)