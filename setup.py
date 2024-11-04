from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="oil-well-calculator",
    version="1.0.0",
    author="Test WL",
    description="A comprehensive oil well analysis tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit==1.37.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "plotly>=5.18.0",
        "scipy>=1.12.0",
        "pyyaml>=6.0.1",
        "scikit-learn==1.5.2",
        "openpyxl>=3.1.2"
    ],
    package_data={
        "": [
            "config/*.yaml",
            "data/equipment_specs/**/*.csv",
            "data/tax_tables/**/*.csv"
        ]
    },
    include_package_data=True,
)
