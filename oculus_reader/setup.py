from setuptools import setup, find_packages

# Find all packages including nested ones
packages = find_packages(where=".", include=["oculus_reader*"])

# Manually add the nested package structure
packages_to_install = ["oculus_reader", "oculus_reader.oculus_reader"]

setup(
    name="oculus_reader_wrapper",
    version="0.1.0",
    packages=packages_to_install,
    package_dir={
        "oculus_reader": ".",
        "oculus_reader.oculus_reader": "oculus_reader",
    },
    install_requires=[
        "numpy",
        "pure-python-adb",
    ],
)
