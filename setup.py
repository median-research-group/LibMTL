from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

requirements = []
with open('requirements.txt', 'r') as f:
    for each in f.readlines():
        requirements.append(each.strip())

setup(
        name='LibMTL',    
        version='1.1.6',   
        description='A PyTorch Library for Multi-Task Learning',   
        author='Baijiong Lin',  
        author_email='bj.lin.email@gmail.com',  
        url='https://github.com/median-research-group/LibMTL',      
        packages=find_packages(),   
        license='MIT',
        platforms=["all"],
        classifiers=['Intended Audience :: Developers',
                     'Intended Audience :: Education',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.8',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Topic :: Scientific/Engineering :: Mathematics',
                     'Topic :: Software Development :: Libraries',],
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=requirements
)

