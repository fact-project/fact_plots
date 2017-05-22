from setuptools import setup

setup(
    name='fact_plots',
    version='0.0.2',
    description='a collection of plotting scrtipts for data of FACT',
    url='https://github.com/fact-project/fact_plots',
    author='Jens Buss',
    author_email='jens.buss@tu-dortmund.de',
    license='BEER',
    packages=[
        'fact_plots',
        'fact_plots.scripts',
    ],
    install_requires=[
        'pandas',           # in anaconda
        'numpy',            # in anaconda
        'matplotlib>=1.4',  # in anaconda
        'python-dateutil',  # in anaconda
        'pytz',             # in anaconda
        'tables',           # needs to be installed by pip for some reason
        # 'hdf5',
        'click',
        'pyfact>=10.4',
        'h5py',
        'scipy',
        'click',
        'docopt',
        'datetime',
        'numexpr',
        'pytest', # also in  conda
        'matplotlib-hep==0.1.0'
    ],
   zip_safe=False,
   entry_points={
    'console_scripts': [
        'fact_plot_data_mc_compare = fact_plots.scripts.plot_data_mc_compare:main',
        'fact_plot_effective_area = fact_plots.scripts.plot_effective_area:main',
        'fact_plot_ped_std_mean_cureent_mean = fact_plots.scripts.plot_ped_std_mean_cureent_mean:main',
        'fact_plot_theta = fact_plots.scripts.plot_theta_squared:main',
    ],
  }
)
