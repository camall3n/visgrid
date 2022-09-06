import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = (req[0] for req in map(lambda x: x.split('#'), f.readlines()))
    install_requires = [req for req in map(str.strip, install_requires) if req]

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

setuptools.setup(
    name="visgrid",
    author="Cameron Allen",
    author_email="csal@brown.edu",
    version="0.0.1",
    python_requires=">=3.9",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    url="https://github.com/camall3n/visgrid",
    description="RL environments for quickly running image-based grid-world experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
    ],
)
