from setuptools import setup

setup(
   name='shapleyx',
   version='0.2',
   description='Python implementation RS-HDMR',
   keywords='sobol sensitivity analysis',
   author='Frederick Bennett',
   author_email='frederick.bennett@des.qld.gov.au',
   packages=['shapleyx'],  #same as name
   install_requires=[
            'numpy',
            'pandas',
            'matplotlib',
            'scipy'
    ]
)