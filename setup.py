# setup.py
import setuptools

# Function to read the contents of requirements.txt
def get_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        # Read lines, strip whitespace, and filter out empty lines/comments
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
    return lines

setuptools.setup(
    name="SABV",
    version="0.1.0",
    author="AquaCoder0010",
    author_email="aquacoder00100011@gmail.com",
    description="Signature Agnostic Binary Visualizer based on SAGMAD",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aquacoder0010/SABV",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
