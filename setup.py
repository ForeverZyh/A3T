from setuptools import find_packages, setup

setup(
    name='a3t',
    packages=find_packages(),
    version='0.0.0',
    description='Implementation for A3T',
    author="Yuhao Zhang",
    license="MIT",
    test_suite="tests",
    install_requires=["forbiddenfruit", "torch>=1.5", "tensorflow>=2.0", "tensorflow-datasets", "nltk>=3.4"],
    # build_requires=["absl-py<0.11,>=0.9"],
    author_email="yuhaoz@cs.wisc.edu",
    long_description="A3T is an adversarial training technique that combines the strengths of augmentation and abstraction techniques. The key idea underlying A3T is to decompose the perturbation space into two subsets, one that can be explored using augmentation and one that can be abstracted.",
    url="https://github.com/ForeverZyh/A3T",
    project_urls={
        "Bug Tracker": "https://github.com/ForeverZyh/A3T/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
