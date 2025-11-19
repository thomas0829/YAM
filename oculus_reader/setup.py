# from setuptools import setup


# setup(
#     name='oculus_reader',
#     version='1.0.0',
#     packages=['oculus_reader'],
#     license='Apache-2.0 License',
#     long_description=open('README.md').read(),
#     install_requires=[
#         'numpy', 'pure-python-adb'
#     ],
#     package_data={'': ['APK/teleop-debug.apk']},
#     include_package_data=True,
# )

from setuptools import setup, find_packages

setup(
    name='oculus_reader',
    version='1.0.0',
    packages=find_packages(),  # finds all packages under this folder
    license='Apache-2.0 License',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy', 'pure-python-adb'
    ],
    package_data={'': ['APK/teleop-debug.apk']},
    include_package_data=True,
)
