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
        'torch==2.4.1',               # Core PyTorch library
        'torchvision==0.19.1',          # For image datasets and transforms
        'joblib==1.4.2',               # For saving/loading models and data
        'scikit-learn==1.5.2',         # For metrics and data preprocessing
        'openai',               # For LLM interaction
        'numpy==2.0.2',                # NumPy for array handling
        'pandas==2.2.2',               # For data manipulation
        'pytz==2024.2',                 # Timezone handling if needed
        'matplotlib==3.9.2',           # For plotting (optional, if visualizations are needed)
        'tqdm==4.66.5',                  # For progress bars during training
        'requests==2.32.3',
        'httpx==0.28.1',
        'streamlit==1.41.1',
         'litellm',
        'google-genai',
        'anthropic==0.49.0',
        'google-adk>=0.1.0'
    ],
    entry_points={
        'console_scripts': [
            'torch_optimize=src.cli:select_model_cli',  # Entry point for CLI
        ],
    },
)


