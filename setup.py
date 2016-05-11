from setuptools import setup

setup(
    name='fact-plotting',
    version='0.0.1',
    description='a collection of plotting scrtipts for data of FACT',
    url='https://github.com/jebuss/fact-plotting',
    author='Jens Buss',
    author_email='jens.buss@tu-dortmund.de',
    license='BEER',
    packages=[
        'fact_plotting',
    ],
    # dependency_links = ['git+https://github.com/mackaiver/gridmap.git#egg=gridmap'],
    package_data={
        # 'erna': ['resources/*'],
    },
    install_requires=[
        'pandas',           # in anaconda
        'numpy',            # in anaconda
        'matplotlib>=1.4',  # in anaconda
        'python-dateutil',  # in anaconda
        'pytz',             # in anaconda
        'tables',           # needs to be installed by pip for some reason
        # 'hdf5',
        'click',
        'docopt',
        'datetime',
        'numexpr',
        'IPython',
        'pytest', # also in  conda
        # 'gridmap>=0.13.1', install from https://github.com/mackaiver/gridmap'
    ],
   zip_safe=False,
   entry_points={
    'console_scripts': [
        'fact_plot_data_mc_compare = fact_plotting.scripts.plot_data_mc_compare:main',
    ],
  }
)
