import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xgboost-deploy",
    version="0.0.2",
    author="Matthew Burke",
    author_email="matthew.wesley.burke@gmail.com",
    description="Deploy XGBoost models in pure python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwburke/xgboost-python-deploy",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
