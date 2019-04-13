from setuptools import setup, find_packages
import os

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='ca_utils',
      version='0.2',
      description='ca utils',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/janclemenslab/ca_utils',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      py_modules=['ca_utils'],
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['numpy', 'scipy', 'h5py', 'scanimage-tiff-reader>=1.4'],
      tests_require=['nose'],
      test_suite='nose.collector',
      include_package_data=True,
      zip_safe=False
      )
