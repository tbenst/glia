from setuptools import setup

setup(name='neurolib',
      version='0.1',
      description='Library for working with MEA data',
      url='https://github.com/tbenst/neuro',
      author='Tyler Benster',
      author_email='tbenst@gmail.com',
      # license='None',
      packages=['neurolib'],
      zip_safe=False,
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',

          # Pick your license as you wish (should match "license" above)

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
      keywords='neuroscience mea microelectrode',
      install_requires=[
          'numpy',  'matplotlib', 'pytest', 'h5py', 'warnings'
      ],
      )
