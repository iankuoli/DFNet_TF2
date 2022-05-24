
from setuptools import setup, find_packages
import os

NAME = 'DFNet-TF2'
VERSION = '0.1'
REQUIRED_PACKAGES = [
    'absl-py==0.8.0',
    'astor==0.8.0',
    'cachetools==3.1.1',
    'certifi==2019.9.11',
    'chardet==3.0.4',
    'cloudpickle==1.1.1',
    'cycler==0.10.0',
    'decorator==4.4.0',
    'gast==0.2.2',
    'google-api-python-client==1.7.11',
    'google-auth==1.6.3',
    'google-auth-httplib2==0.0.3',
    'google-auth-oauthlib==0.4.1',
    'google-pasta==0.1.7',
    'grpcio==1.24.0',
    'h5py==2.10.0',
    'httplib2==0.14.0',
    'idna==2.8',
    'joblib==0.14.0',
    'Keras==2.3.0',
    'Keras-Applications==1.0.8',
    'Keras-Preprocessing==1.1.0',
    'kiwisolver==1.1.0',
    'Markdown==3.1.1',
    'matplotlib==3.1.1',
    'numpy==1.17.2',
    'oauthlib==3.1.0',
    'opencv-python==4.1.1.26',
    'opt-einsum==3.1.0',
    'pandas==0.25.1',
    'Pillow==6.2.0',
    'protobuf==3.9.2',
    'pyasn1==0.4.7',
    'pyasn1-modules==0.2.6',
    'pyparsing==2.4.2',
    'python-dateutil==2.8.0',
    'pytz==2019.2',
    'PyYAML==5.1.2',
    'requests==2.22.0',
    'requests-oauthlib==1.2.0',
    'rsa==4.0',
    'scikit-learn==0.21.3',
    'scipy==1.3.1',
    'six==1.12.0',
    'tensorboard==2.0.0',
    'tensorflow==2.6.4',
    'tensorflow-estimator==2.0.0',
    'termcolor==1.1.0',
    'tfp-nightly==0.9.0.dev20191003',
    'tqdm==4.36.1',
    'uritemplate==3.0.0',
    'urllib3==1.25.6',
    'Werkzeug==0.16.0',
    'wrapt==1.11.2'
]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='My training application package.'
    )
