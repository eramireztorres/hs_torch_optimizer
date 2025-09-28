import os
import pandas as pd
import warnings
import joblib
from typing import Optional, List, Dict, Any, Callable
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool, agent_tool
from google.adk.models.lite_llm import LiteLlm
import subprocess
from google.adk.agents import Agent


def fix_target_column(
    input_path: str,
    target_column: Optional[str] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ensure the target column is the last column for hs_optimize.

    Steps:
    1. Load the file from `input_path` (CSV or Excel).
    2. Determine `target_column`: if `None`, use the last column; if an integer string, interpret as column index.
    3. Reorder columns so that the target is last.
    4. Write the fixed data to `output_path` (overwrites original if None).
    5. Return metadata including row count, feature names, and column types.
    """
    was_read_as_csv = False
    try:
        # Load CSV or Excel
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(input_path)
            was_read_as_csv = True
        elif ext in ['.xls', '.xlsx']:
            try:
                engine = 'xlrd' if ext == '.xls' else 'openpyxl'
                df = pd.read_excel(input_path, engine=engine)
            except Exception as e:
                warnings.warn(f"Could not read Excel file ({e}), attempting to read as CSV.")
                df = pd.read_csv(input_path)
                was_read_as_csv = True
        else:
            # Try to read as CSV as a last resort for files with no extension or unknown extension
            try:
                df = pd.read_csv(input_path)
                was_read_as_csv = True
                warnings.warn(f"Unsupported file type '{ext}', but successfully read as CSV.")
            except Exception as e:
                 warnings.warn(f"Unsupported file type '{ext}' and failed to read as CSV: {e}")
                 return {"error": f"Unsupported file type: {ext}"}

        # Infer target column
        if target_column is None:
            tc = df.columns[-1]
        else:
            try:
                idx = int(target_column)
                tc = df.columns[idx]
            except (ValueError, KeyError, IndexError):
                tc = str(target_column)

        if tc not in df.columns:
            warnings.warn(
                f"Target column '{tc}' not found in columns {list(df.columns)}. "
                f"Defaulting to last column '{df.columns[-1]}'."
            )
            tc = df.columns[-1]

        # Reorder columns: features first, then target
        features = [c for c in df.columns if c != tc]
        ordered_cols = features + [tc]
        df = df[ordered_cols]

        # Write output
        final_output_path = output_path
        if final_output_path is None:
            if was_read_as_csv:
                # If we read a mis-named file as CSV, save it as CSV
                final_output_path = os.path.splitext(input_path)[0] + '.csv'
            else:
                final_output_path = input_path

        os.makedirs(os.path.dirname(final_output_path) or '.', exist_ok=True)

        output_ext = os.path.splitext(final_output_path)[1].lower()
        if was_read_as_csv or output_ext == '.csv':
            df.to_csv(final_output_path, index=False)
        elif output_ext in ['.xls', '.xlsx']:
            # Writing to .xls is deprecated and might require another library.
            # Let's default to .xlsx if .xls is requested for writing.
            if output_ext == '.xls':
                warnings.warn("Writing to .xls format is deprecated. Saving as .xlsx instead.")
                final_output_path = os.path.splitext(final_output_path)[0] + '.xlsx'
            df.to_excel(final_output_path, index=False, engine='openpyxl')
        else:
            # If no extension on output path, assume csv
            df.to_csv(final_output_path, index=False)


        # Prepare metadata
        metadata: Dict[str, Any] = {
            "fixed_path": final_output_path,
            "n_rows": int(df.shape[0]),
            "n_features": int(len(features)),
            "feature_columns": features,
            "target_column": tc,
            "column_types": {col: str(df[col].dtype) for col in df.columns},
        }
        return metadata

    except Exception as e:
        warnings.warn(f"An unexpected error occurred in fix_target_column: {e}")
        return {"error": str(e)}


def fix_target_column_tool():
    return FunctionTool(func=fix_target_column)








def read_model_history(
    history_path: str,
    index: Optional[int] = None
) -> Dict[str, Any]:
    """
    Read a torch_optimize .joblib history file and optionally fetch a specific entry.

    - If index is None, returns a summary listing number of entries and global metrics for each.
    - If index is provided, returns the model_code and metrics dict for that entry.

    Does not raise errors but emits warnings and returns best-effort defaults.

    Returns:
      {
        'n_models': int,
        'summaries': List[Dict[str, Any]]  # only if index is None
        'model_code': str,                 # only if index provided
        'metrics': Dict[str, Any]          # only if index provided
      }
    """
    if not os.path.exists(history_path):
        warnings.warn(f"History file '{history_path}' not found.")
        return {'n_models': 0, 'summaries': [] if index is None else None}

    try:
        history = joblib.load(history_path)
    except Exception as e:
        warnings.warn(f"Failed to load joblib file '{history_path}': {e}")
        return {'n_models': 0, 'summaries': [] if index is None else None}

    if not isinstance(history, list):
        warnings.warn(f"Expected a list of model entries, got {type(history).__name__}.")
        return {'n_models': 0, 'summaries': [] if index is None else None}

    n = len(history)
    result: Dict[str, Any] = {'n_models': n}

    # Summaries: global_metrics for each entry
    if index is None:
        summaries: List[Dict[str, Any]] = []
        for i, entry in enumerate(history):
            if not isinstance(entry, dict):
                warnings.warn(f"Entry {i} is not a dict; skipping.")
                continue
            metrics = entry.get('metrics', {})
            global_metrics = metrics.get('global_metrics', metrics)
            summaries.append({'index': i, 'global_metrics': global_metrics})
        result['summaries'] = summaries
        return result

    # Specific entry
    if not isinstance(index, int) or index < 0 or index >= n:
        warnings.warn(f"Index {index} out of range [0, {n-1}]. Defaulting to 0.")
        index = 0

    entry = history[index]
    if not isinstance(entry, dict):
        warnings.warn(f"Entry at index {index} is not a dict. Returning empty structure.")
        return {'n_models': n, 'model_code': None, 'metrics': None}

    model_code = entry.get('model_code')
    if model_code is None:
        warnings.warn(f"No 'model_code' found in entry {index}.")

    metrics = entry.get('metrics')
    if metrics is None:
        warnings.warn(f"No 'metrics' found in entry {index}.")

    result['model_code'] = model_code
    result['metrics'] = metrics
    return result


# Register as a FunctionTool for the root agent
def read_model_history_tool():
    return FunctionTool(func=read_model_history)


# Default LLM for code generation
DEFAULT_MODEL = LiteLlm(
    model=os.environ.get("LLM_MODEL", "openai/gpt-5-nano")
)

# --- SUB-AGENT: Model Code Generator ---
# Generates Python code files defining a load_model function per CLI contract
model_code_generator = LlmAgent(
    name="ModelCodeGenerator",
    model=DEFAULT_MODEL,
    instruction="""
You are a Model Code Generator. Given a summary of desired model characteristics and dataset context, write a complete Python file that defines a `load_model` function. Follow these rules exactly:
1. For scikit-learn models:
   - Signature: `def load_model():`
   - Inside, import the needed estimator and return an untrained instance with default or specified hyperparameters.
2. For neural networks:
   - Signature: `def load_model(X_train, y_train, **kwargs):`
   - Include imports, reproducibility seed, define a PyTorch `nn.Module`, and return `model, optimizer, criterion`.
3. Do not include any top-level code other than imports inside `load_model`.
4. Output only the code (no commentary or markdown).
""",
    description="LLM-based generator of `load_model` code for torch_optimizer",
    output_key="generated_code",
)
# Wrap as a tool for the root agent to invoke
code_generator_tool = agent_tool.AgentTool(agent=model_code_generator)

# --- TOOL: Save Generated Model Code to File ---

def save_model_code(file_path: str, code: str) -> Dict[str, str]:
    """
    Write the provided Python `code` to `file_path` and return the file path.
    """
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(code)
    return {"file_path": file_path}

save_model_code_tool = FunctionTool(func=save_model_code)




def run_torch_optimize(
    data: str,
    model: Optional[str] = None,
    model_provider: Optional[str] = None,
    history_file_path: Optional[str] = None,
    iterations: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    extra_info: Optional[str] = None,
    is_regression: Optional[str] = None,
    metrics_source: Optional[str] = None,
    error_model: Optional[str] = None,
    initial_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invokes the `torch_optimize` command-line tool with the specified parameters.

    This function constructs and executes a subprocess call to the `torch_optimize`
    CLI, capturing its output. It serves as a Python interface to the underlying
    optimization script, allowing it to be called programmatically.

    Args:
        data (str): Path to the dataset file or directory.
        model (Optional[str], optional): The LLM model to use for optimization.
            Defaults to None.
        model_provider (Optional[str], optional): The provider of the LLM.
            Defaults to None.
        history_file_path (Optional[str], optional): Path to store the optimization
            history. Defaults to None.
        iterations (Optional[int], optional): Number of optimization iterations.
            Defaults to None.
        epochs (Optional[int], optional): Number of training epochs per iteration.
            Defaults to None.
        batch_size (Optional[int], optional): Batch size for training.
            Defaults to None.
        lr (Optional[float], optional): Learning rate for training. Defaults to None.
        extra_info (Optional[str], optional): Additional context for the LLM.
            Defaults to None.
        is_regression (Optional[str], optional): Specifies if the task is regression
            ("true" or "false"). Defaults to None.
        metrics_source (Optional[str], optional): The data source for metrics
            ('validation' or 'test'). Defaults to None.
        error_model (Optional[str], optional): The LLM model for error correction.
            Defaults to None.
        initial_model_path (Optional[str], optional): Path to an initial model file.
            Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing the execution results:
            - 'exit_code' (int): The exit code of the process.
            - 'stdout' (str): The standard output.
            - 'stderr' (str): The standard error.
            - 'history_file_path' (str): The path to the history file.
    """
    # Base command
    cmd = ['torch_optimize', '--data', data]

    # Optional arguments
    if model is not None:
        cmd += ['--model', model]
    if model_provider is not None:
        cmd += ['--model-provider', model_provider]
    if history_file_path is not None:
        cmd += ['--history-file-path', history_file_path]
    if iterations is not None:
        cmd += ['--iterations', str(iterations)]
    if epochs is not None:
        cmd += ['--epochs', str(epochs)]
    if batch_size is not None:
        cmd += ['--batch-size', str(batch_size)]
    if lr is not None:
        cmd += ['--lr', str(lr)]
    if extra_info is not None:
        cmd += ['--extra-info', extra_info]
    if is_regression is not None:
        cmd += ['--is-regression', is_regression]
    if metrics_source is not None:
        cmd += ['--metrics-source', metrics_source]
    if error_model is not None:
        cmd += ['--error-model', error_model]
    if initial_model_path is not None:
        cmd += ['--initial-model-path', initial_model_path]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
    except FileNotFoundError:
        warnings.warn("`torch_optimize` CLI not found in PATH.")
        return {'exit_code': -1, 'stdout': '', 'stderr': 'CLI not found', 'history_file_path': history_file_path}
    except Exception as e:
        warnings.warn(f"Failed to run torch_optimize: {e}")
        return {'exit_code': -1, 'stdout': '', 'stderr': str(e), 'history_file_path': history_file_path}

    if proc.returncode != 0:
        warnings.warn(
            f"torch_optimize exited with code {proc.returncode}. stderr: {proc.stderr}"
        )

    return {
        'exit_code': proc.returncode,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
        'history_file_path': history_file_path
    }


# Register as a FunctionTool for the root agent
def run_torch_optimize_tool():
    return FunctionTool(func=run_torch_optimize)

root_agent = Agent(
    name="TorchOptimizeCoordinator",
    model=DEFAULT_MODEL,
    description=(
        "Orchestrates the full torch_optimize workflow: "
        "preprocess input, run optimization, "
        "analyze results, and optionally generate new model code."
    ),
    instruction="""
You are the TorchOptimizeCoordinator. Given a user request to optimize a model, follow these steps:

1. **Ensure input is ready**  
   - The `torch_optimize` command expects the target column to be the last column in the dataset.
   - If the user provides a dataset file (like CSV or Excel) and either explicitly asks to prepare the file OR specifies a target column by name or index, you should use the `fix_target_column_tool` tool. This tool will place the target column at the end and return the path to the corrected file.
   - If the user doesn't specify a target column, you can assume the provided file is already correctly formatted and pass it directly to `run_torch_optimize`.
   - If the user provided pre-split Joblib or CSVs, you can also skip this step.

2. **Run optimization**  
   - Call `run_torch_optimize` with the appropriate `--data` argument (the fixed file, split-joblib, or folder), plus model, iterations, epochs, batch_size, lr, extra-info, etc., based on user flags.

3. **Inspect history**  
   - Call `read_model_history` with no index to get a summary of `global_metrics`.  
   - If the best metric meets the user’s goal (e.g., accuracy ≥ threshold), report success.

4. **Iterate if needed**  
   - If no entry is satisfactory, ask the user for desired changes or automatically invoke `code_generator_tool` to produce new `load_model()` code.  
   - Save it via `save_model_code_tool`, then call `run_torch_optimize` again on that code.

5. **Return**  
   - Provide the final `history_file_path`, key metrics, and any generated model code file path.

Use these tools if needed. Always confirm critical arguments before invoking a tool.
""",
    tools=[
        fix_target_column_tool(),
        run_torch_optimize_tool(),
        read_model_history_tool(),
        code_generator_tool,
        save_model_code_tool,
    ],
)
