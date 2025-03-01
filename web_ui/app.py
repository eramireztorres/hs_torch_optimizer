import streamlit as st
import os
import subprocess
import time
import tempfile


def set_env_variable(key, value):
    os.environ[key] = value

    if os.name == 'nt':
        os.system(f'setx {key} "{value}"')
    else:
        bashrc_path = os.path.expanduser('~/.bashrc')
        with open(bashrc_path, 'a') as f:
            f.write(f'\nexport {key}="{value}"\n')
            
st.set_page_config(
    page_title="Torch Model Optimizer",
    page_icon="🔥",
    layout="wide"
)


# Load existing environment variables if they exist
existing_openai_key = os.getenv("OPENAI_API_KEY", "")
existing_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
existing_gemini_key = os.getenv("GEMINI_API_KEY", "") 
existing_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

st.sidebar.header("⚙️ API Key Settings")

# Input for OpenAI API Key (pre-populated if already set)
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key:",
    value=existing_openai_key,
    type="password"
)


# New: Input for Gemini API Key (pre-populated if already set)
gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    value=existing_gemini_key,
    type="password"
)

# New: Input for Anthropic API Key (pre-populated if already set)
anthropic_api_key = st.sidebar.text_input(
    "Enter your Anthropic API Key:",
    value=existing_anthropic_key,
    type="password"
)


# Input for OpenRouter API Key (pre-populated if already set)
openrouter_api_key = st.sidebar.text_input(
    "Enter your OpenRouter API Key:",
    value=existing_openrouter_key,
    type="password"
)


# Button to save API keys
if st.sidebar.button("Save API Keys"):
    if openai_api_key:
        set_env_variable("OPENAI_API_KEY", openai_api_key)
        st.sidebar.success("OpenAI API Key saved successfully!")
    if openrouter_api_key:
        set_env_variable("OPENROUTER_API_KEY", openrouter_api_key)
        st.sidebar.success("OpenRouter API Key saved successfully!")
    if gemini_api_key:
        set_env_variable("GEMINI_API_KEY", gemini_api_key)
        st.sidebar.success("Gemini API Key saved successfully!")
    if anthropic_api_key:
        set_env_variable("ANTHROPIC_API_KEY", anthropic_api_key)
        st.sidebar.success("Anthropic API Key saved successfully!")
    if not (openai_api_key or openrouter_api_key or gemini_api_key or anthropic_api_key):
        st.sidebar.warning("Please enter at least one API key to save.")


def run_cli_with_output(data_path, args_dict, output_placeholder):
    command = ['torch_optimize', '--data', data_path]

    for key, value in args_dict.items():
        if value is not None and value != "None":
            cli_key = f'--{key.replace("_", "-")}'
            command.append(cli_key)
            command.append(str(value))

    output_placeholder.code(' '.join(command), language='bash')

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        output_placeholder.text(output)
        time.sleep(0.1)

    process.stdout.close()
    process.wait()
    output_placeholder.text(output)
    st.success("Optimization completed!")


def construct_full_paths(history_file_path, input_data_folder, directory_path=None, uploaded_files=None):
    def is_base_filename(path):
        return os.path.basename(path) == path

    if directory_path:
        base_dir = directory_path
    elif uploaded_files:
        base_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(base_dir, exist_ok=True)
    else:
        base_dir = input_data_folder

    if is_base_filename(history_file_path):
        history_file_path = os.path.join(base_dir, history_file_path)

    return history_file_path


def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    return file_paths

def convert_is_regression(value):    
    if value == "None":
        return None
    if value.lower() == "true":
        return "true"
    return "false"



left_col, right_col = st.columns([2, 3])

with left_col:
    st.header("📂 Upload Data Files")

    uploaded_files = st.file_uploader(
        "Drag and drop or browse your file (.joblib, .csv).",
        type=["joblib", "csv"],
        accept_multiple_files=False
    )

    directory_path = st.text_input("Enter a directory path (outputs will be saved here if provided, otherwise in 'outputs/' folder):")

    if uploaded_files:
        st.success("1 file uploaded.")
        st.write(f"📄 {uploaded_files.name}")
    elif directory_path:
        if os.path.isdir(directory_path):
            st.success(f"Directory selected: {directory_path}")
        else:
            st.error("Invalid directory path. Please enter a valid path.")
    else:
        st.info("Please upload at least one file or enter a directory path.")

    st.header("⚙️ Configure Optimization Parameters")

    model = st.text_input("Model:", value='gpt-4o-mini')
    history_file_path = st.text_input("History Model/Metrics File Path:", value='model_history.joblib')
    iterations = st.number_input("Number of Iterations:", min_value=1, max_value=100, value=10)
    extra_info = st.text_area("Extra Info for the LLM:", value='Not available')
    batch_size = st.number_input("Batch Size:", min_value=1, value=32)
    lr = st.number_input("Learning Rate:", min_value=0.0001, format="%.5f", value=0.001)
    epochs = st.number_input("Epochs:", min_value=1, value=10)
    is_regression = st.selectbox("Is Regression?", options=["None", "true", "false"], index=0)
    metrics_source = st.selectbox("Metrics Source:", options=["validation", "test"], index=0)
    error_model = st.text_input("Error Model (Optional):")


with right_col:
    st.header("🖥️ Output")

    output_placeholder = st.empty()

    if st.button("Run Optimization"):
        if uploaded_files:
            saved_files = save_uploaded_files([uploaded_files])
            data_path = saved_files[0]
            input_data_folder = os.path.dirname(data_path)
        elif directory_path:
            data_path = directory_path
            input_data_folder = directory_path
        else:
            st.error("Please upload a file or enter a directory path before running.")
            st.stop()

        history_file_path = construct_full_paths(
            history_file_path,
            input_data_folder,
            directory_path,
            uploaded_files
        )

        st.write(f"Output file path: {history_file_path}")

        args_dict = {
            'model': model,
            'history_file_path': history_file_path,
            'iterations': iterations,
            'extra_info': extra_info,
            'batch_size': batch_size,
            'lr': lr,
            'epochs': epochs,            
            'metrics_source': metrics_source,
            'error_model': error_model
        }
        
        is_regression_value = convert_is_regression(is_regression)        
       
        if is_regression_value is not None:            
            args_dict['is_regression'] = is_regression_value


        run_cli_with_output(data_path, args_dict, output_placeholder)
        st.success(f"Optimization completed! Outputs saved in: {os.path.dirname(history_file_path)}")
