# Neural Network Model Optimizer Project

This project aims to optimize neural network models using PyTorch by iterating through model training, evaluation, and improvements suggested by an LLM (Large Language Model). The system dynamically improves neural network architectures and hyperparameters based on input data and model performance history.

## Features
- Supports classification and regression tasks.
- Handles both flat feature vectors and image inputs (2D arrays).
- Utilizes LLM to suggest improvements in model architecture and hyperparameters.
- Dynamically applies improvements using hot-swapping techniques.
- Saves model history and evaluation metrics.
  
## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/eramireztorres/hs_torch_optimizer.git
    cd hs_torch_optimizer
    ```

2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages:**
    Run the following command to install dependencies:
    ```bash
    pip install .
    ```

## Export the API keys of your models

Before using, make sure to export your OpenAI API key as an environment variable. 

Linux or macOS:

```bash
export OPENAI_API_KEY='your_api_key_here'
```

Or in windows:

```bash
setx OPENAI_API_KEY=your_api_key_here
```

## Run the App as CLI with Options

You can run the torch_optimize command-line interface (CLI) with several options for customizing the optimization process. 
Make sure the joblib data file contains a Python dictionary with the keys 'X_train', 'y_train', 'X_test', and 'y_test'. 
The application uses 'y_train' data to determine whether it is a classification or regression problem.

### Usage

torch_optimize [-h] --data DATA [--history-file-path HISTORY_FILE_PATH] [--model MODEL] [--iterations ITERATIONS] [--extra-info EXTRA_INFO] [--epochs EPOCHS]

### Optional Arguments

- **`-h, --help`**:  
  Show the help message and exit.

- **`--data DATA`, `-d DATA`**:  
  Path to a `.joblib` file containing training and test data. The file should include a dictionary with keys like `'X_train'`, `'y_train'`, `'X_test'`, and `'y_test'`. These should be NumPy arrays representing the feature and target datasets for model training and evaluation.

- **`--history-file-path HISTORY_FILE_PATH`, `-hfp HISTORY_FILE_PATH`**:  
  Path to the `.joblib` file where the model history will be saved. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is `'model_history.joblib'`.

- **`--model MODEL`, `-m MODEL`**:  
  The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Defaults to `'gpt-4o'`.

- **`--iterations ITERATIONS`, `-i ITERATIONS`**:  
  The number of iterations to run. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is `5`.

- **`--extra-info EXTRA_INFO, -ei EXTRA_INFO`**:  
  Additional context or information to provide to the LLM for more informed suggestions. Examples include class imbalance, noisy labels, or outlier data. Default is 'Not available'.

- **`--epochs EPOCHS, -e EPOCHS`**:  
  Number of epochs to train the neural network in each iteration. Default is `10`.

### Example 1

Here’s an example of how to run the app with custom data, model history path, iterations, and epochs:

```bash
torch_optimize -d my_classification_data.joblib -hfp output_model_history.joblib -i 10 --epochs 20
```

### Example 2

Example with Class Imbalance for Classification:

```bash
torch_optimize -d my_classification_data.joblib -hfp classification_history.joblib -i 10 --epochs 15 --extra-info "Binary classification with class imbalance, 4:1 ratio between class 0 and class 1."
```

In this case, the application will pass the additional information to the LLM, which can then suggest using custom loss functions or class weighting techniques to address the class imbalance.

## License
[MIT](LICENSE)