from setuptools import setup, find_packages

setup(
    name="companion_app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "pydantic",
        "sentence-transformers",
        "crawl4ai",
        "asyncio",
        "python-multipart",
        "googlesearch-python",
        "duckduckgo-search",
        "aiohttp",
        "beautifulsoup4",
        "litellm"
    ],
    python_requires=">=3.8",
) 