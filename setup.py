from pathlib import Path
from setuptools import find_packages, setup


setup(
    name="edagraph",
    version="1.0.0",
    description="EDA-Graph: Graph Signal Processing of Electrodermal Activity for Emotional States Detection.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Luis R. Mercado-Diaz, Yedukondala Rao Veeranki, Fernando Marmolejo-Ramos, Hugo F. Posada-Quintero",
    url="https://github.com/jouninlrmd/eda-graph",
    packages=find_packages(exclude=("tests", "scripts")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "scikit-learn>=1.2",
        "networkx>=3.0",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "joblib>=1.2",
        "PyYAML>=6.0",
        "statsmodels>=0.14",
        "scikit-posthocs>=0.8",
        "tqdm>=4.65",
    ],
    extras_require={"eda": ["neurokit2>=0.2.7"]},
    entry_points={
        "console_scripts": [
            "edagraph-extract = scripts.extract_features:main",
            "edagraph-classify = scripts.run_classification:main",
            "edagraph-stats = scripts.run_statistics:main",
        ],
    },
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
