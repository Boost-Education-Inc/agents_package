from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='agents',
    version='0.1',
    description='Agents package',
    author='Boost Education',
    author_email='juan.sebastian@boostedu.co',
    packages=['boostEdu'],
    include_package_data=True,
    install_requires=requirements,
)