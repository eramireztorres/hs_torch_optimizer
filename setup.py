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
        '': ['prompts/*.txt'],  # Include all .txt files in the prompts folder
    },
    install_requires=[
        'torch',               # Core PyTorch library
        'torchvision',          # For image datasets and transforms
        'joblib',               # For saving/loading models and data
        'scikit-learn',         # For metrics and data preprocessing
        'openai',               # For LLM interaction
        'numpy',                # NumPy for array handling
        'pandas',               # For data manipulation
        'pytz',                 # Timezone handling if needed
        'matplotlib',           # For plotting (optional, if visualizations are needed)
        'tqdm'                  # For progress bars during training
    ],
    entry_points={
        'console_scripts': [
            'torch_optimize=src.cli:select_model_cli',  # Entry point for CLI
        ],
    },
)



# from setuptools import setup, find_packages

# setup(
#     name='hs-model-optimizer',
#     version='1.0.0',
#     description='A project for hot-swapping model optimization using LLM suggestions.',
#     author='Erick Eduardo Ramirez Torres',
#     author_email='erickeduardoramireztorres@gmail.com',
#     packages=find_packages(),
#     include_package_data=True,  # Ensure package data is included
#     package_data={
#         '': ['prompts/*.txt'],  # Include all .txt files in prompts folder
#     },
#     install_requires=[
#         'scikit-learn',
#         'xgboost',
#         'joblib',
#         'openai',
#         'lightgbm',
#         'pytz'
#     ],
#     entry_points={
#         'console_scripts': [
#             'hs_optimize=src.cli:select_model_cli'  
#         ],
#     },
# )

