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

## Usage

This application can be used in three different ways:

1.  **ADK (Agent Development Kit):** For natural language interaction with an AI agent team.
2.  **Streamlit Web UI:** A user-friendly graphical interface for guided optimization.
3.  **CLI (Command-Line Interface):** For scripted and automated workflows.

---

### 1. ADK (Agent Development Kit)

Interact with the optimizer through a natural language chat interface powered by the ADK. This mode allows you to converse with an AI agent team to discuss and implement model improvements.

**Prerequisites:**
- This mode requires an OpenAI model. Ensure you have your OpenAI API key set as an environment variable:
  ```bash
  export OPENAI_API_KEY='your_openai_api_key_here'
  ```

**Launching the ADK:**
```bash
cd adk
adk web
```
This command starts a local web server for the chat interface.

---

### 2. Streamlit Web UI

A user-friendly web interface built with Streamlit for a more guided experience.

**Launching the Web UI:**
```bash
cd web_ui
streamlit run app.py
```
This will launch the application in your default web browser.

**Configuring API Keys in the Web UI:**
1.  Open the sidebar (click the 🌟 icon).
2.  Enter your API keys for OpenAI, OpenRouter, Gemini, or Anthropic.
3.  Click "Save API Keys".

**Using the Web UI:**
1.  **Upload Data:** Drag and drop `.joblib` or `.csv` files, or provide a directory path.
2.  **Configure Parameters:** Set the LLM model, iterations, epochs, etc.
3.  **Run Optimization:** Click "Run Optimization" to start.
4.  **Review Results:** View the output and saved model history location.

---

### 3. CLI (Command-Line Interface)

Run the optimizer from the command line for scripted and automated workflows.

**Prerequisites:**
- Export the API key for your chosen model provider as an environment variable (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).

**Usage:**
```bash
torch_optimize [OPTIONS]
```

**Arguments:**
- `--data, -d`: Path to the dataset.
- `--history-file-path, -hfp`: Path to save model history.
- `--model, -m`: LLM model to use (e.g., `'gpt-4o-mini'`).
- `--is-regression, -ir`: Specify `true` for regression tasks.
- `--metrics-source, -ms`: `validation` (default) or `test`.
- `--iterations, -i`: Number of optimization iterations.
- `--error-model, -em`: LLM for error correction.
- `--extra-info, -ei`: Additional context for the LLM.
- `--epochs, -e`: Training epochs per iteration.

**Example:**
```bash
torch_optimize -d my_data.joblib -hfp model_history.joblib -i 10 -m gpt-4o
```

### 4. Supported Input File Formats

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
   - `'X'` (features),  
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


## License
[MIT](LICENSE)