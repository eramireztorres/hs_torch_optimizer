# LLM-Powered Neural Network Model Optimizer Project

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

### OpenAI Models

For OpenAI models, export your API key as an environment variable:

Linux or macOS:

```bash
export OPENAI_API_KEY='your_openai_api_key_here'
```

Or in windows:

```bash
setx OPENAI_API_KEY "your_openai_api_key_here"
```

### Other Models via OpenRouter

To use Llama, Gemini or other models with OpenRouter, follow these steps:

1. Visit [OpenRouter](https://openrouter.ai/) and log in or create an account.
2. Navigate to the API keys section in your account dashboard and generate a new API key.
3. Export the API key as an environment variable:
   - For Linux or macOS:
     ```bash
     export OPENROUTER_API_KEY='your_openrouter_api_key_here'
     ```
   - For Windows:
     ```bash
     setx OPENROUTER_API_KEY "your_openrouter_api_key_here"
     ```

## Run the App as CLI with Options

You can run the torch_optimize command-line interface (CLI) with several options for customizing the optimization process. 
Make sure the joblib data file contains a Python dictionary with the keys 'X_train', 'y_train', 'X_test', and 'y_test'. 
The application uses 'y_train' data to determine whether it is a classification or regression problem.

### Usage

torch_optimize [-h] --data DATA [--history-file-path HISTORY_FILE_PATH] [--model MODEL] [--iterations ITERATIONS] [--extra-info EXTRA_INFO] [--epochs EPOCHS]

### Supported Input File Formats

The application supports the following input formats:

1. **Pre-split `.joblib` file**:  
   A Python dictionary containing the keys:  
   - `'X_train'`, `'y_train'` (training data),  
   - `'X_test'`, `'y_test'` (test data).

2. **Pre-split `.csv` files**:  
   A directory containing the following files:  
   - `X_train.csv`, `y_train.csv`, `X_test.csv`, and `y_test.csv`.

3. **Unsplit `.joblib` file**:  
   A Python dictionary containing the keys:  
   - `'X'` (features: it can be an array of 2-D numpy arrays for the image case),  
   - `'y'` (targets).  
   The application will create a validation split from the data (default split ratio is 80/20).

4. **Unsplit `.csv` files**:  
   A directory containing two files:  
   - `X.csv` (features),  
   - `y.csv` (targets).  
   The application will create a validation split.

5. **Single `.csv` file**:  
   A single CSV file where:  
   - All columns except the last are treated as features (`X`),  
   - The last column is assumed to be the target (`y`).

### Arguments

- **`-h, --help`**:  
  Show the help message and exit.

- **`--data DATA`, `-d DATA`**:  
  Path to the input dataset. Supported formats are .joblib files, directories containing .csv files, or a single .csv file.
  The application handles validation splits automatically for unsplit datasets.
- **`--history-file-path HISTORY_FILE_PATH`, `-hfp HISTORY_FILE_PATH`**:  
  Path to the `.txt` or `.joblib` file where the model history will be saved. The history includes models, their hyperparameters, and performance metrics for each iteration. Default is `'model_history.joblib'`.

- **`--model MODEL`, `-m MODEL`**:  
  The name of the LLM model to use for generating suggestions and improvements for models and hyperparameters. Examples include `'gpt-4'`, `'llama-3.1'`. Defaults to `'gpt-4o-mini'`.

- **`--metrics-source METRICS_SOURCE`, `-ms METRICS_SOURCE`**:  
  Specify the source of the metrics to show to the LLM:
  - **`validation`** (default): Metrics are computed on a validation split created from the training data.
  - **`test`**: Metrics are computed on the test data.

- **`--iterations ITERATIONS`, `-i ITERATIONS`**:  
  The number of iterations to run. Each iteration involves training a model, evaluating its performance, and generating improvements. Default is `10`.

- **`--extra-info EXTRA_INFO, -ei EXTRA_INFO`**:  
  Additional context or information to provide to the LLM for more informed suggestions. Examples include:
  - **Class imbalance**: Disparity in the number of samples per class.
  - **Noisy labels**: Incorrect or inconsistent labels in the dataset.
  - **Outliers**: Unusual or extreme data points in the features or targets.  
  Default is `'Not available'`.

- **`--epochs EPOCHS, -e EPOCHS`**:  
  Number of epochs to train the neural network in each iteration. Default is `10`.


### Example 1

Here’s an example of how to run the app with custom data, model history path, iterations, and epochs:

```bash
torch_optimize -d my_classification_data.joblib -hfp output_model_history.joblib -i 10 --epochs 20 -m gpt-4o
```

### Example 2

Example with Class Imbalance for Classification:

```bash
torch_optimize -d my_classification_data.joblib -hfp classification_history.joblib -i 10 --epochs 15 --extra-info "Binary classification with class imbalance, 4:1 ratio between class 0 and class 1."
```

In this case, the application will pass the additional information to the LLM, which can then suggest using custom loss functions or class weighting techniques to address the class imbalance.

## License
[MIT](LICENSE)