from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8')

setup(
    name='shapleyx',
    version='0.2',
    description='Python implementation RS-HDMR',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='sobol sensitivity analysis',
    author='Frederick Bennett',
    author_email='frederick.bennett@des.qld.gov.au',
    license='MIT',
    packages=find_packages(include=['shapleyx', 'shapleyx.*']),
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy'
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'mypy>=0.910',
            'flake8>=3.9',
            'jupyter>=1.0',
            'ipython>=7.0'
        ],
        'docs': [
            'mkdocs>=1.2',
            'mkdocs-material>=7.0'
        ]
    },
    project_urls={
        'Source': 'https://github.com/frbennett/shapleyx',
        'Documentation': 'https://github.com/frbennett/shapleyx/tree/main/docs',
        'Bug Reports': 'https://github.com/frbennett/shapleyx/issues'
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Operating System :: OS Independent'
    ]
)