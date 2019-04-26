import os
from setuptools import setup, find_packages


setup(
    name="libenv",
    version="0.1",
    packages=find_packages(),
    install_requires=["gym~=0.10.0", "cffi~=1.11", "numpy~=1.14"],
    extras_require={"dev": ["pytest", "pytest-benchmark"]},
    use_scm_version=False
    if os.environ.get("USE_SCM_VERSION", "1") == "0"
    else {"root": "..", "relative_to": __file__},
    setup_requires=["setuptools_scm"],
)
