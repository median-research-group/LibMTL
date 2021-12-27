from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
        name='LibMTL',    
        version='1.0',   
        description='A PyTorch Library for Multi-Task Learning',   
        author='Baijiong Lin',  
        author_email='linbj@mail.sustech.edu.cn',  
        url='https://github.com/median-research-group/LibMTL',      
        packages=find_packages(),   
        license='MIT',
        platforms=["all"],
        classifiers=['Programming Language :: Python :: 3.8'],
        long_description=long_description,
        long_description_content_type='text/markdown',
)