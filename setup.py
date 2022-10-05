import os

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

__version__ = None
PACKAGE_NAME = "tritony"
with open(os.path.join(HERE, "{0}/version.py".format(PACKAGE_NAME))) as fr:
    exec(fr.read())

with open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = """
tritonclient[all]>=2.18.0
protobuf>=3.5.0,<3.20
orjson>=3.6.8
reretry>=0.11.1
""".strip().split(
    "\n"
)
dev_extra_requires = """
pytest
pre-commit
""".strip().split(
    "\n"
)

data_files = [
    ("", ["LICENSE"]),
]

setup(
    name=PACKAGE_NAME,
    version=__version__,
    license="BSD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rtzr/tritony",
    keywords=[
        "grpc",
        "http",
        "triton",
        "tensorrt",
        "inference",
        "server",
        "service",
        "client",
        "nvidia",
        "rtzr",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    author="Arthur",
    author_email="arthur@rtzr.ai",
    description="Tiny configuration for Triton Inference Server",
    packages=find_packages(
        where=".",
        include=[
            PACKAGE_NAME,
        ],
        exclude=["tests"],
    ),
    install_requires=install_requires,
    extras_require={"dev": dev_extra_requires},
    zip_safe=False,
    data_files=data_files,
)
