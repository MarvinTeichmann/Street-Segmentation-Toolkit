try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'sst',
    'version': '0.1.0',
    'author': 'Martin Thoma, Marvin Teichman, Sebastian Bittel, Vitali Kaiser',
    'author_email': 'info@martin-thoma.de',
    'packages': ['sst', 'sst.networks'],
    'scripts': ['bin/sst'],
    'package_data': {'sst': ['templates/*', 'misc/*', 'static/*']},
    # 'url': 'https://github.com/MartinThoma/pyspell',
    # 'license': 'MIT',
    'description': 'Street Segmentation Tools',
    'long_description': ("This package contains multiple tools which can "
                         "be used to classify the pixels of a given image "
                         "into 'street'/'no street'."),
    'install_requires': [
        "argparse",
        # "theano",
        # "nose"
    ],
    'keywords': ['autonomous driving', 'cognition'],
    # 'download_url': 'https://github.com/MartinThoma/pyspell',
    'classifiers': ['Development Status :: 1 - Planning',
                    'Environment :: Console',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: MIT License',
                    'Natural Language :: English',
                    'Programming Language :: Python :: 2.7',
                    'Programming Language :: Python :: 3',
                    'Topic :: Scientific/Engineering :: Artificial Intelligence',
                    'Topic :: Software Development',
                    'Topic :: Utilities'],
    'zip_safe': False,
    'test_suite': 'nose.collector'
}

setup(**config)
