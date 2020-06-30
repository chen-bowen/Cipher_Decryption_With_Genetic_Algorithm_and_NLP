from setuptools import setup

setup(
    name="cipher_decryption",
    version="1.0",
    description="Application of NLP and Genetic Algorithm in cipher decryption",
    author="Bowen Chen",
    packages=["cipher_decryption"],  # same as name
    install_requires=[
        "pandas",
        "numpy",
        "ipykernel",
    ],  # external packages as dependencies
)
