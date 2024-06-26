from setuptools import setup, find_packages

setup(name='glia',
      version='0.5.2',
      description='Elegant support functions for Neuroscientists.',
      url='https://github.com/tbenst/glia',
      author='Tyler Benster',
      author_email='tbenst@gmail.com',
      # license='None',
      packages=find_packages(),
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
          'Programming Language :: Python :: 3.6',
      ],
      keywords='neuroscience mea microelectrode',
      entry_points={
          'console_scripts': [
            'glia=glia_scripts.command_line:main',
            'glia-profile=glia_scripts.profile:main',
          ],
      },
      setup_requires=['pytest-runner', 'setuptools_scm'],
      install_requires=[
          "av",
          "beautifulsoup4",
          "click",
          "elephant",
          "h5py",
          "matplotlib",
          "neo",
          "numba",
          "numpy",
          "pandas",
          "pyarrow",
          "pyret",
          "pytest",
          "pyYAML",
          "requests",
          "scikit-learn",
          "scipy",
          "seaborn",
          "tables",
          "tqdm",
          "webcolors",
          "websockets"
      ],
      include_package_data=True,
      tests_require=['pytest'],
      )
