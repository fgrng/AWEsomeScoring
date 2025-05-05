#!/usr/bin/env python3
"""
Setup script for AWEsomeScoring
"""

from setuptools import setup, find_packages

setup(
    name="awesomescoring",
    version="0.1.0",
    description="A command line tool for automated writing evaluation (AWE) using different AI models",
    author="Fabian Grünig",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/AWEsomeScoring",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.8.0",
        "mistralai>=0.0.7",
        "pyyaml>=6.0",
        "requests>=2.30.0",
    ],
    entry_points={
        'console_scripts': [
            'awescore=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.7',
)
