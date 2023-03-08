from setuptools import find_packages,setup
from typing import List
def get_requirements()-> List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] =[]
    """
    Write a code to read requirements.txt and append each requirement in requirement_list variable
    """
    return requirement_list



setup(
     name="Anurag_Koushal_APPINVENTIVE",
     version="0.0.1",
     author="Anurag_Koushal",
     author_email="anuragkoushal1993@gmail.com",
     packages = find_packages(),
     install_requires=get_requirements(),
     )