from setuptools import setup


setup(
    name="libenv",
    version="0.0.5",
    packages=["libenv"],
    install_requires=["gym~=0.10", "cffi~=1.11", "numpy~=1.14"],
    extras_require={"dev": ["pytest", "pytest-benchmark"]},
)
