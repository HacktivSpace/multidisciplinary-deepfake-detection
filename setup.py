from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multidisciplinary-deepfake-detection",
    version="0.1.0",
    author="HacktivSpace",
    author_email="devsupport@hacktivspace.com",
    description="A multidisciplinary deepfake detection system using images, audios, and videos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HacktivSpace/multidisciplinary-deepfake-detection",
    project_urls={
        "Bug Tracker": "https://github.com/HacktivSpace/multidisciplinary-deepfake-detection/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.2.4",
        "tensorflow>=2.4.1",
        "torch>=1.8.1",
        "scikit-learn>=0.24.2",
        "librosa>=0.8.0",
        "opencv-python>=4.5.1.48",
        "matplotlib>=3.3.4",
        "seaborn>=0.11.1",
        "nltk>=3.5",
        "spacy>=3.0.6",
        "joblib>=1.0.1",
        "flask>=1.1.2",
        "gunicorn>=20.1.0",
        "psycopg2-binary>=2.8.6",
        "python-dotenv>=0.17.0",
    ],
    entry_points={
        "console_scripts": [
            "run-app=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
