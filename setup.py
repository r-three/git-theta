from setuptools import setup

setup(
    name="git_cml",
    version="0.0.0",
    description="Version control system for model checkpoints.",
    author="Colin Raffel",
    author_email="craffel@gmail.com",
    url="https://github.com/r-three/checkpoint-vcs",
    packages=["git_cml"],
    scripts=["bin/git-cml", "bin/git-cml-filter"],
    long_description="Version control system for model checkpoints.",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Version Control",
    ],
    keywords="git vcs machine-learning",
    license="MIT",
    install_requires=[
        "GitPython",
        "torch",
    ],
)
