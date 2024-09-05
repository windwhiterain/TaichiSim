from setuptools import setup,find_packages

setup(
   name='TaichiSim',
   version='0.1',
   description='A modular, extensible, physic library using taichi.',
   author='whitingyan',
   author_email='1712428442@qq.com',
   packages=find_packages(),  
   install_requires=['taichi'], 
)