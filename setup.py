from setuptools import setup, find_packages

setup(
    name='hs-torch-optimizer',
    version='1.0.0',
    description='A project for hot-swapping neural network model optimization using LLM suggestions with PyTorch.',
    author='Erick Eduardo Ramirez Torres',
    author_email='erickeduardoramireztorres@gmail.com',
    packages=find_packages(),
    # packages=find_packages(where='src'),
    # package_dir={'': 'src'},  # This tells setuptools to look for packages inside the 'src' folder

    include_package_data=True,  # Ensure package data is included
    package_data={
        'src': ['prompts/*.txt'],  # Include all .txt files in the src/prompts folder
        'src.dynamic_models': ['*.py'],  # Include dynamic model template files
    },
    install_requires=[
        'torch',               # Core PyTorch library
        'torchvision',          # For image datasets and transforms
        'joblib==1.4.2',               # For saving/loading models and data
        'scikit-learn==1.5.2',         # For metrics and data preprocessing
        'openai',               # For LLM interaction
        'numpy>=1.24',                # NumPy for array handling
        'pandas>=2.2',               # For data manipulation
        'pytz>=2024.2',                 # Timezone handling if needed
        'matplotlib>=3.9',           # For plotting (optional, if visualizations are needed)
        'tqdm>=4.66.5',                  # For progress bars during training
        'requests>=2.32.3',
        'httpx>=0.28.1',
        'streamlit>=1.41.1',
         'litellm',
        'google-genai',
        'anthropic>=0.49.0',
        'google-adk',
         'xlsxwriter',
        'xlrd'

    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'pytest-timeout>=2.1.0',
            'coverage>=7.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'pytest-timeout>=2.1.0',
            'coverage>=7.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'isort>=5.12.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'torch_optimize=src.cli:select_model_cli',  # Entry point for CLI
        ],
    },
)


