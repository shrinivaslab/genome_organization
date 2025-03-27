from setuptools import setup, find_packages

setup(
    name="genome_organization",
    version="0.1.0",
    packages=find_packages(where="src") + ['polykit', 'polykit.renderers'],
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "requests"
    ],
) 