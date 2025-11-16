from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-similarity-toolkit",
    version="0.2.0",
    author="MiniMax Agent",
    author_email="your.email@example.com",
    description="A professional toolkit for image similarity comparison using deep learning models with database support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-similarity-toolkit",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/image-similarity-toolkit/issues",
        "Documentation": "https://github.com/yourusername/image-similarity-toolkit/blob/main/docs/usage.md",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "clip": [
            "ftfy",
            "regex",
        ],
    },
    entry_points={
        "console_scripts": [
            "image-similarity=image_similarity.cli:main",
        ],
    },
)
