from setuptools import setup, find_packages

setup(
    name="stringpullkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.23,<2",
        "scipy>=1.10,<2",   
        "pandas>=2.0,<3",
        "matplotlib>=3.7,<4",
        "seaborn",
        "h5py",
        "xlsxwriter",
        "opencv-python",
        "Pillow",
        "opencv"
],
    extras_require={
        "dlc": ["deeplabcut"],
    },
    entry_points={
        "console_scripts": [
            "stringpullkit=stringpullkit.__main__:main",
        ],
    },
    python_requires=">=3.9",
)
