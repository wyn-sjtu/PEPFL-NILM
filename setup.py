"""Package configuration compatible with Python 3.8."""
from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="film-adaptive-huber-nilm",
    version="0.1.0",
    description="Fixed 1D CNN with private FiLM and residual-adaptive Huber for federated NILM",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"nilm_f": ["resources/*.json"]},
    python_requires=">=3.8",
    install_requires=["torch>=2.0,<3", "numpy>=1.24,<2", "pandas>=2.0,<3", "scikit-learn>=1.3,<2"],
    entry_points={"console_scripts": ["nilm-f=nilm_f.cli:main"]},
)
