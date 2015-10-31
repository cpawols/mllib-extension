from setuptools import setup

setup(name='reduct_feature_selection',
      version='0.1',
      description='package for paralell feature selection and discretization',
      url='https://github.com/cpawols/mllib-extension',
      author='Pawel Olszewski, Krzysztof Rutkowski',
      author_email='cpawols@gmail.com, krisun17@gmail.com',
      license='free',
      packages=['reduct_feature_selection'],
      install_requires=[
          'markdown',
      ],
      zip_safe=False)
