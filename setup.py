from distutils.core import setup
import shutil
import os

source = "configs"
destination = "evaluation/configs"
dest_was_created = False
if not os.path.exists(destination):
    dest_was_created = True
    shutil.copytree(source, destination)

try:
    setup(
        name="transaction_metrics",
        version="0.0.2",
        packages=['tmetrics', 'tmetrics.preprocess', 'tmetrics.pipelines', 'tmetrics.models'],
        package_dir = {'tmetrics': 'evaluation'},
        package_data={'tmetrics': ['configs/*/*.yaml']},
        install_requires=[
            "pandas==2.2.0",
            "pyspark==3.5.0",
            "seaborn==0.13.2",
            "optuna==3.5.0",
            "torcheval==0.0.7",
            "omegaconf==2.3.0",
            "lightgbm==4.3.0",
<<<<<<< HEAD
            "ebes==0.0.4",
=======
            "mamba_ssm==1.2.2",
            "torchcde==0.2.5",
            "ebes==0.0.3",
>>>>>>> 2ee8e228027c1951528dfb72efce7bfd153ac6aa
            "sdmetrics==0.15.1"
        ],
    )
except Exception:
    pass
finally:
    if dest_was_created:
        shutil.rmtree(destination)