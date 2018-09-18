from setuptools import setup, find_packages

with open('requirements.txt') as f:
        INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(
      name='hip-tensorflow',
      version='0.1',
      description='HIP model in TensorFlow',
      url='https://github.com/kasrsf/hip-tensorflow',
      author='Kasra Safari',
      author_email='kasrasafari@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      zip_safe=False
)