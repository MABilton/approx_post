from setuptools import setup, find_packages

setup(
   name='approx_post',
   version='0.0.0',
   author='Matthew Bilton',
   author_email='Matt.A.Bilton@gmail.com',
   python_requires='>=3.6',
   packages=find_packages(exclude=('examples')),
   url="https://github.com/MABilton/approx_post",
   install_requires=[
      'jax>=0.2.19',
      'jaxlib>=0.1.69',
      'numpy>=1.19.5'
      ]
)