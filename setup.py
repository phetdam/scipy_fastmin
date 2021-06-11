# setup.py to build scipy_fastmin C extension modules.

from numpy import get_include
from setuptools import setup, Extension

# package name and summary/short description
_PACKAGE_NAME = "scipy_fastmin"
_PACKAGE_SUMMARY = """Multivariate optimizers implemented in C extension \
modules compatible with scipy.optimize.minimize.\
"""
# extra compilation arguments for extension modules
_EXTRA_COMPILE_ARGS = ["-std=gnu11"]


def _get_ext_modules():
    """Returns a list of setuptools.Extension modules to build.

    Returns
    -------
    list
        List of Extension instances to be sent to ext_modules kwargs of setup.
    """
    # list is incomplete, there will be other extensions to build
    return [
        Extension(
            name="utils.c_utils",
            sources=[f"{_PACKAGE_NAME}/utils/c_utils.c"],
            include_dirs=[f"{_PACKAGE_NAME}/include", get_include()],
            extra_compile_args=_EXTRA_COMPILE_ARGS
        )
    ]


def _setup():
    # get version
    with open("VERSION") as vf:
        version = vf.read().strip()
    # get long description from README.rst
    with open("README.rst") as rf:
        long_desc = rf.read().strip()
    # run setuptools
    setup(
        name=_PACKAGE_NAME,
        version=version,
        description=_PACKAGE_SUMMARY,
        long_description=long_desc,
        long_description_content_type="text/x-rst",
        author="Derek Huang",
        author_email="djh458@stern.nyu.edu",
        license="MIT",
        url="https://github.com/phetdam/scipy_fastmin",
        packages=[_PACKAGE_NAME],
        python_requires=">=3.6",
        install_requires=["numpy>=1.19.1", "scipy>=1.5.2"],
        ext_package=_PACKAGE_NAME,
        ext_modules=_get_ext_modules()
    )


if __name__ == "__main__":
    _setup()