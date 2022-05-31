import os
import sys
from shutil import rmtree

from setuptools import Command, find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

__version__ = None
PACKAGE_NAME = "tritony"
with open(os.path.join(HERE, "{0}/version.py".format(PACKAGE_NAME))) as fr:
    exec(fr.read())

install_requires = """
tritonclient[all]~=2.18.0
protobuf>=3.5.0,<3.20
more-itertools~=8.13.0
orjson==3.6.8
""".strip().split(
    "\n"
)
dev_extra_requires = """
pytest
pre-commit
""".strip().split(
    "\n"
)


# brought from https://github.com/navdeep-G/setup.py
class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            os.system("rm -vrf ./*.egg-info")
            rmtree(os.path.join(HERE, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/* -r rtzr")

        self.status("Creating git tag")
        os.system(f"git tag -a -f -m '' v{__version__}")

        self.status("Pushing git tags…")
        os.system(f"git push -v origin refs/tags/v{__version__}")

        sys.exit()


setup(
    name=PACKAGE_NAME,
    version=__version__,
    url="",
    license="modified MIT",
    author="Arthur",
    author_email="arthur@rtzr.ai",
    description="",
    packages=find_packages(
        where=".",
        include=[
            PACKAGE_NAME,
        ],
        exclude=["test"],
    ),
    install_requires=install_requires,
    extras_require={"dev": dev_extra_requires},
    zip_safe=False,
    include_package_data=True,
    package_data={},
    cmdclass={
        "deploy": UploadCommand,
    },
)
# rtzr
