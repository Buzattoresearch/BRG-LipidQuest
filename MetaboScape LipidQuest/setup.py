from setuptools import setup, find_packages

setup(
    name="LipidQuest",
    version="0.1",
    packages=find_packages(),
    install_requires=["pandas", "openpyxl"],
    entry_points={
        "console_scripts": [
            "LipidQuest = LipidQuest:main",
        ],
    },
)
