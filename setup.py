from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8')

setup(
    name='shapleyx',
    version='0.5.1',
    description='Global sensitivity analysis with RS-HDMR — Sobol, Shapley, PAWN, and more',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='shapley sobol sensitivity analysis hdmr uncertainty quantification',
    author='Frederick Bennett',
    author_email='frederick.bennett@des.qld.gov.au',
    license='MIT',
    packages=find_packages(include=['shapleyx', 'shapleyx.*']),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-learn',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'mypy>=0.910',
            'flake8>=3.9',
            'jupyter>=1.0',
            'ipython>=7.0',
        ],
        'docs': [
            'mkdocs>=1.2',
            'mkdocs-material>=7.0',
            'mkdocstrings[python]',
            'mkdocs-jupyter',
            'pymdown-extensions',
        ],
    },
    project_urls={
        'Source': 'https://github.com/frbennett/shapleyx',
        'Documentation': 'https://frbennett.github.io/shapleyx/',
        'Bug Reports': 'https://github.com/frbennett/shapleyx/issues',
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Operating System :: OS Independent',
    ],
)