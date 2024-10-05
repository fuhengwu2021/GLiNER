from setuptools import setup, find_packages

# Read the content of the README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gliner",
    version=None,  # The dynamic version will be fetched from gliner.__version__
    description="Generalist model for NER (Extract any entity types from texts)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois",
    maintainer="Urchade Zaratiana",
    url="https://github.com/urchade/GLiNER",
    packages=find_packages(include=["gliner", "gliner.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.38.2",
        "huggingface_hub>=0.21.4",
        "tqdm",
        "onnxruntime",
        "sentencepiece",
    ],
    keywords=[
        "named-entity-recognition",
        "ner",
        "data-science",
        "natural-language-processing",
        "artificial-intelligence",
        "nlp",
        "machine-learning",
        "transformers",
    ],
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
