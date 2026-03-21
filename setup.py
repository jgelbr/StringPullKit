from setuptools import setup, find_packages

setup(
    name="stringpullkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.2.2",
        "scipy>=1.15.3",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "seaborn",
        "h5py",
        "xlsxwriter",
        "opencv-python",
        "Pillow",
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
