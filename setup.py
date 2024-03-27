# from distutils.core import setup
from setuptools import find_packages, setup
import os

# Read requirements file
with open('requirements.txt', 'r') as fh:
    _requires = fh.read().splitlines()

def scan_dir(path, prefix=None):
    if prefix is None:
        prefix = path

    # Scan resources package for files to include
    file_list = []
    for root, dirs, files in os.walk(path):
        # Strip this part as setup wants relative directories
        root = root.replace(prefix, '')
        root = root.lstrip('/\\')

        for filename in files:
            if filename[0:8] == '__init__':
                continue
            file_list.append(os.path.join(root, filename))

    return file_list


# Determine the extra resources and scripts to pack
resources_list = scan_dir('./aip/data')

setup(
    name="aip",
    version="0.3",
    description="""TM11005 Week 3 Machine Learning Python Package""",
    license="Apache 2.0 License",
    author="Martijn P. A. Starmans, Jose M. Castillo T.",
    author_email="m.starmans@erasmusmc.nl",
    include_package_data=True,
    package_data={"aip.data": resources_list},
    packages=[
        "aip",
        "aip.data",
        "aip.data.segmentation",
        "aip.data.registration",
        "aip.data.ML",
        "aip.data.registration.Brain",
        "aip.data.registration.Fundus"
    ],
    install_requires=_requires
)
