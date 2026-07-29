from setuptools import find_packages, setup


setup(
    name="distillalign",
    version="0.1.0",
    description="Autoregressive video distillation and teacher-normalized distribution evaluation",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "alignment-eval=distribution_eval.cli:main",
        ]
    },
)
