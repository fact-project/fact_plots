from setuptools import setup, find_packages

setup(
    name='fact_plots',
    version='0.0.6',
    description='a collection of plotting scrtipts for data of FACT',
    url='https://github.com/fact-project/fact_plots',
    author='Kai Brügge, Jens Buss, Maximilian Nöthe',
    author_email='jens.buss@tu-dortmund.de',
    license='BEER',
    packages=find_packages(),
    install_requires=[
        'click',
        'docopt',
        'h5py',
        'matplotlib-hep==0.1.0',
        'matplotlib>=1.5',  # in anaconda
        'numexpr',
        'numpy',            # in anaconda
        'pandas',           # in anaconda
        'pyfact>=0.14.1',
        'pytest',
        'python-dateutil',  # in anaconda
        'pytz',             # in anaconda
        'scipy',
        'tables',           # needs to be installed by pip for some reason
        'tqdm',
    ],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'fact_plot_data_mc_compare = fact_plots.scripts.plot_data_mc_compare:main',
            'fact_plot_effective_area = fact_plots.scripts.plot_effective_area:main',
            'fact_plot_excess_rate = fact_plots.scripts.plot_excess_rate:main',
            'fact_plot_ped_std_mean_curent_mean = fact_plots.scripts.plot_ped_std_mean_curent_mean:main',
            'fact_plot_theta_squared = fact_plots.scripts.plot_theta_squared:main',
            'fact_plot_energy_migration = fact_plots.scripts.plot_energy_migration:main',
            'fact_plot_bias_resolution = fact_plots.scripts.plot_bias_resolution:main',
            'fact_plot_angular_resolution = fact_plots.scripts.plot_angular_resolution:main',
            'fact_plot_skymap = fact_plots.scripts.plot_skymap:main',
        ],
    }
)
