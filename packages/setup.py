from setuptools import setup, find_packages

# To create the package:
# 1: Use 'pip install build' in this root folder.
# 2: Go to packages folder and use 'python -m build'.

# To install the project:
# 1: Go to dist folder and use 'pip install titanic_ml_project-0.1-py3-none-any.whl --force-reinstall'.

setup(
    name='titanic_ml_project',
    version='0.1',
    description='ML project around Titanic dataset',
    author='Luis Cerdeño Mota',
    author_email='luiscerdenomota@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'mlflow',
        'lightgbm',
        'seaborn',
        'openpyxl'
    ],
    entry_points={
        'console_scripts': [
            'titanic_ml_project=src.train_eval:main'
        ]
    }
)
