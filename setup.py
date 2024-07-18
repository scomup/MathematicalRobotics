from setuptools import setup, find_packages

setup(
    name='mathR',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pyqtgraph',
        'pyopengl',
        'scikit-sparse',
        'libsuitesparse-dev'
        ],
    entry_points={},
)