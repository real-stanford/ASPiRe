from distutils.core import setup

setup(name='ASPiRe',
      version='0.0.1',
      packages=['ASPiRe'],
      install_requires=[
          'pytorch',
          'stable-baselines3',
          'wandb',
          'gym',
          'numpy',
          'contextlib',
      ])
