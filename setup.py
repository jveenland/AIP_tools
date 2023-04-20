from distutils.core import setup

# Read requirements file
with open('requirements.txt', 'r') as fh:
    _requires = fh.read().splitlines()

setup(
    name="aip",
    version="0.3",
    description="""TM11005 Week 3 Machine Learning Python Package""",
    license="Apache 2.0 License",
    author="Martijn P. A. Starmans, Jose M. Castillo T.",
    author_email="m.starmans@erasmusmc.nl",
    package_dir={"": "aip"},
    include_package_data=True,
    package_data={"aip": ["*.dcm", "*.nii.gz", "*.csv"]},
    packages=[
        "aip"
    ],
    install_requires=_requires
)
