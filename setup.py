
from setuptools import setup, find_packages

setup(
    name='pdf-extractor',
    version='0.1',
    packages=find_packages(exclude=['client*']),
    license='MIT',
    description='A tool for keesing report face data extraction',
    long_description=open('README.txt').read(),
    install_requires=['PyMuPDF'],
    url='https://github.com/tawsinDOTuddin/pdf-extractor',
    author='Tawsin Uddin Ahmned',
    author_email='tawsin.uddin@gmail.com'
)
