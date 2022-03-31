import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thesis-code",
    version="0.0.1",
    author="Sebastian Griesbach",
    author_email="sebastian.griesbach@student.uni-tuebingen.de",
    description="All code modules created during my thesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sebastian-Griesbach/rl-parameter_based_value_functions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    include_package_data = True,
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)