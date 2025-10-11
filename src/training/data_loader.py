import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    @staticmethod
    def load_data(input_path):
        """
        Load data from a directory or file and standardize it into a dictionary format.

        Args:
            input_path (str): Path to the input file or directory.

        Returns:
            dict: A dictionary containing 'X_train', 'y_train', 'X_test', and 'y_test'.

        Raises:
            ValueError: If the input is invalid or unsupported.
        """
        input_path = Path(input_path)

        if input_path.is_file():
            return DataLoader._load_from_file(input_path)
        elif input_path.is_dir():
            return DataLoader._load_from_directory(input_path)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    @staticmethod
    def _load_from_csv(file_path):
        df = pd.read_csv(file_path)
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(df.columns):
            return {
                "X_train": df["X_train"].to_numpy(),
                "y_train": df["y_train"].to_numpy(),
                "X_test": df["X_test"].to_numpy(),
                "y_test": df["y_test"].to_numpy(),
            }
        else:
            raise ValueError(
                "CSV file must contain 'X_train', 'y_train', 'X_test', and 'y_test' columns."
            )

    @staticmethod
    def _load_from_json(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(data.keys()):
            return data
        else:
            raise ValueError(
                "JSON file must contain 'X_train', 'y_train', 'X_test', and 'y_test' keys."
            )

    @staticmethod
    def _load_from_file(file_path):
        if file_path.suffix == ".joblib":
            data = joblib.load(file_path)
            return DataLoader._handle_data_split(data)
        elif file_path.suffix == ".csv":
            return DataLoader._handle_csv_file(file_path)
        elif file_path.suffix in [".xls", ".xlsx"]:
            return DataLoader._handle_excel_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    @staticmethod
    def _load_from_directory(directory_path):
        """
        Load data from a directory, handling cases with pre-split or unsplit CSV files.
        """
        files = {f.stem: f for f in directory_path.iterdir() if f.suffix == ".csv"}

        if {"X_train", "y_train", "X_test", "y_test"}.issubset(files.keys()):
            X_train = pd.read_csv(files["X_train"])
            y_train = pd.read_csv(files["y_train"]).squeeze()
            X_test = pd.read_csv(files["X_test"])
            y_test = pd.read_csv(files["y_test"]).squeeze()

            categorical_cols = X_train.select_dtypes(include=["object"]).columns

            if len(categorical_cols) > 0:
                print(f"Encoding categorical columns: {list(categorical_cols)}")
                X_train = DataLoader._encode_categorical(X_train, categorical_cols)
                X_test = DataLoader._encode_categorical(
                    X_test, categorical_cols, fit=False
                )

            if y_train.dtype == "object" or isinstance(y_train.iloc[0], str):
                print("Encoding categorical target labels in y_train.")
                y_train = pd.factorize(y_train)[0]

            if y_test.dtype == "object" or isinstance(y_test.iloc[0], str):
                print("Encoding categorical target labels in y_test.")
                y_test = pd.factorize(y_test)[0]

            data = {
                "X_train": X_train.to_numpy(),
                "y_train": y_train.to_numpy(),
                "X_test": X_test.to_numpy(),
                "y_test": y_test.to_numpy(),
            }
            data["is_pre_split"] = True
            return data

        elif {"X", "y"}.issubset(files.keys()):
            X = pd.read_csv(files["X"])
            y = pd.read_csv(files["y"]).squeeze()

            categorical_cols = X.select_dtypes(include=["object"]).columns
            if len(categorical_cols) > 0:
                print(f"Encoding categorical columns: {list(categorical_cols)}")
                X = DataLoader._encode_categorical(X, categorical_cols)

            return {"X": X, "y": y, "is_pre_split": False}

        raise ValueError(
            "Invalid directory structure: Must contain either ('X_train', 'y_train', 'X_test', 'y_test') or ('X', 'y')."
        )

    @staticmethod
    def _encode_categorical(X, categorical_cols, fit=True):
        """
        Encode categorical features using One-Hot Encoding.

        Args:
            X (pd.DataFrame): Feature dataframe.
            categorical_cols (list): List of categorical column names.
            fit (bool): Whether to fit a new encoder or use an existing one.

        Returns:
            pd.DataFrame: Transformed feature dataframe.
        """
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X[categorical_cols] = X[categorical_cols].fillna("Missing")  # Fill NaN values

        if fit:
            transformed = encoder.fit_transform(X[categorical_cols])
        else:
            transformed = encoder.transform(X[categorical_cols])

        encoded_df = pd.DataFrame(
            transformed,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index,
        )

        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, encoded_df], axis=1)

        return X

    @staticmethod
    def _handle_data_split(data):
        if "X_train" in data and "y_train" in data:
            data["is_pre_split"] = True
            return data
        elif "X" in data and "y" in data:
            return {"X": data["X"], "y": data["y"], "is_pre_split": False}

        else:
            raise ValueError(
                "Input data must contain either ('X_train', 'y_train', 'X_test', 'y_test') or ('X', 'y')."
            )

    @staticmethod
    def _handle_csv_file(file_path):
        df = pd.read_csv(file_path)

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == "object" or isinstance(y.iloc[0], str):
            print("Encoding categorical target labels.")
            y = pd.factorize(y)[0]  # Encode string labels as integers

        for col in X.select_dtypes(include=["number"]).columns:
            X[col] = X[col].fillna(X[col].median())

        categorical_cols = X.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical columns: {list(categorical_cols)}")
            X = DataLoader._encode_categorical(X, categorical_cols)

        X.fillna(0, inplace=True)
        X = X.astype("float32")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        return {
            "X": X if isinstance(X, np.ndarray) else X.to_numpy(),
            "y": y if isinstance(y, np.ndarray) else y.to_numpy(),
            "is_pre_split": False,
        }

    @staticmethod
    def _handle_excel_file(file_path):
        try:
            engine = "xlrd" if file_path.suffix == ".xls" else "openpyxl"
            df = pd.read_excel(file_path, engine=engine)
        except Exception:
            print(
                f"Warning: Could not read file {file_path} as Excel, attempting to read as CSV."
            )
            df = pd.read_csv(file_path)

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == "object" or isinstance(y.iloc[0], str):
            print("Encoding categorical target labels.")
            y = pd.factorize(y)[0]  # Encode string labels as integers

        for col in X.select_dtypes(include=["number"]).columns:
            X[col] = X[col].fillna(X[col].median())

        categorical_cols = X.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical columns: {list(categorical_cols)}")
            X = DataLoader._encode_categorical(X, categorical_cols)

        X.fillna(0, inplace=True)
        X = X.astype("float32")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        return {
            "X": X if isinstance(X, np.ndarray) else X.to_numpy(),
            "y": y if isinstance(y, np.ndarray) else y.to_numpy(),
            "is_pre_split": False,
        }

    @staticmethod
    def _split_data(X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
        if not isinstance(y_test, pd.Series):
            y_test = pd.Series(y_test)

        y_train = y_train.fillna(y_train.median())
        y_test = y_test.fillna(y_test.median())

        return {
            "X_train": (
                X_train if isinstance(X_train, np.ndarray) else X_train.to_numpy()
            ),
            "y_train": y_train.to_numpy(),
            "X_test": X_test if isinstance(X_test, np.ndarray) else X_test.to_numpy(),
            "y_test": y_test.to_numpy(),
        }
