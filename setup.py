from distutils.core import setup
import shutil
import os

package_data = [
    dict(
        source='configs',
        destination="evaluation/configs",
        dest_was_created=False,
    ),
    dict(
        source='data',
        destination="evaluation/data",
        dest_was_created=False,
    )
]

for pack_dt in package_data:
    source = pack_dt['source']
    destination = pack_dt['destination']
    if not os.path.exists(destination):
        pack_dt['dest_was_created'] = True
        shutil.copytree(source, destination)

try:
    setup(
        name="transaction_metrics",
        version="0.0.3",
        packages=['tmetrics', 'tmetrics.preprocess'],
        package_dir = {'tmetrics': 'evaluation'},
        package_data={'tmetrics': ['configs/*/*.yaml', 'data/*/*.json']},
        install_requires=[
            "pandas==2.2.0",
            "pyspark==3.5.0",
            "seaborn==0.13.2",
            "optuna==3.5.0",
            "torcheval==0.0.7",
            "omegaconf==2.3.0",
            "lightgbm==4.3.0",
            "mamba_ssm==1.2.2",
            "torchcde==0.2.5",
            "ebes==0.0.3",
            "sdmetrics==0.15.1"
        ],
    )
except Exception:
    pass
finally:
    for pack_dt in package_data:
        dest_was_created = pack_dt['dest_was_created']
        destination = pack_dt['destination']
        if dest_was_created:
            shutil.rmtree(destination)