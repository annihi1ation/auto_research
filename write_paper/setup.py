from setuptools import setup, find_namespace_packages

setup(
    name="write_paper",
    version="0.1.0",
    packages=find_namespace_packages(include=["src.*"]),
    package_dir={"": "."},
    install_requires=[
        "pydantic-graph>=0.1.0",
        "sentence-transformers>=2.2.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "aiohttp>=3.8.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
)
