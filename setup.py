# coding: utf-8
from pathlib import Path
from setuptools import find_packages, setup

root_path = Path(__file__).resolve().parent

# Get version number
for line in (root_path / "joeynmt/__init__.py").read_text(encoding="utf8").splitlines():
    if line.startswith("__version__"):
        version = line.strip().split('=')[-1].strip()[1:-1]
        break

# Get dependencies
install_requires = (root_path / "requirements.txt").read_text(encoding="utf8")

# Get descriptions
long_description = (root_path / "docs/JoeyNMT_v2.md").read_text(encoding="utf8")

setup(
    name='joeynmt',
    version=version,
    description='Minimalist NMT for educational purposes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jasmijn Bastings and Julia Kreutzer',
    url='https://github.com/joeynmt/joeynmt',
    license='Apache License',
    install_requires=install_requires.splitlines(),
    packages=find_packages(exclude=[]),
    python_requires='>=3.7',
    project_urls={
        'Documentation': 'http://joeynmt.readthedocs.io/en/latest/',
        'Source': 'https://github.com/joeynmt/joeynmt',
        'Tracker': 'https://github.com/joeynmt/joeynmt/issues',
    },
    entry_points={
        'console_scripts': [],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
)
