from setuptools import setup

# Python 3.7.3
REQUIRED_PACKAGES = [
    'numpy >= 1.16.4', 'scipy >= 1.3.0', 'tensorflow >= 1.14.0',
    'Pillow >= 6.0.0', 'scikit-learn >= 0.21.2'
]

setup(name='mlchr',
      version='0.1',
      description='A python library for character pattern recognition',
      url='https://github.com/rizart/mlchr',
      author='Rizart Dona',
      author_email='rizart.dona@gmail.com',
      license='MIT',
      packages=['mlchr'],
      install_requires=REQUIRED_PACKAGES,
      zip_safe=False)
