from setuptools import setup, find_packages

setup(
    name="genome_organization",
    version="0.1.0",
    packages=find_packages() + ['polykit', 'polykit.renderers', 'utilis'],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "requests"
    ],
) 