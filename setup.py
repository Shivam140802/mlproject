from setuptools import setup, find_packages
from typing import List

var='-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    It removes any empty lines and comments.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements=[req.replace('\n', '') for req in requirements]

        if var in requirements:
            requirements.remove(var)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Shivam Gupta',
    author_email='sg4781778@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    )