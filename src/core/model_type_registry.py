"""
Model Type Registry - Strategy Pattern for Model Type Selection

This module implements the Strategy Pattern to eliminate if/else chains
for selecting the appropriate model updater and improver classes based on
task type (regression/classification) and data type (image/tabular).
"""

from typing import Tuple, Type
from enum import Enum


class TaskType(Enum):
    """Enum for task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DataType(Enum):
    """Enum for data types."""

    TABULAR = "tabular"
    IMAGE = "image"


class ModelTypeRegistry:
    """
    Registry for model updater and improver classes based on task and data type.

    This class uses the Strategy Pattern to dynamically select the appropriate
    model components without requiring if/else chains.
    """

    def __init__(self):
        """Initialize the registry with empty mappings."""
        self._updater_registry = {}
        self._improver_registry = {}

    def register_updater(
        self, task_type: TaskType, data_type: DataType, updater_class: Type
    ):
        """
        Register a model updater class for a specific task and data type.

        Args:
            task_type: The task type (classification or regression)
            data_type: The data type (tabular or image)
            updater_class: The updater class to register
        """
        key = (task_type, data_type)
        self._updater_registry[key] = updater_class

    def register_improver(
        self, task_type: TaskType, data_type: DataType, improver_class: Type
    ):
        """
        Register an LLM improver class for a specific task and data type.

        Args:
            task_type: The task type (classification or regression)
            data_type: The data type (tabular or image)
            improver_class: The improver class to register
        """
        key = (task_type, data_type)
        self._improver_registry[key] = improver_class

    def get_updater_class(self, task_type: TaskType, data_type: DataType) -> Type:
        """
        Get the appropriate updater class for the given task and data type.

        Args:
            task_type: The task type
            data_type: The data type

        Returns:
            The updater class

        Raises:
            KeyError: If no updater is registered for the given combination
        """
        key = (task_type, data_type)
        if key not in self._updater_registry:
            raise KeyError(
                f"No updater registered for task_type={task_type.value}, "
                f"data_type={data_type.value}"
            )
        return self._updater_registry[key]

    def get_improver_class(self, task_type: TaskType, data_type: DataType) -> Type:
        """
        Get the appropriate improver class for the given task and data type.

        Args:
            task_type: The task type
            data_type: The data type

        Returns:
            The improver class

        Raises:
            KeyError: If no improver is registered for the given combination
        """
        key = (task_type, data_type)
        if key not in self._improver_registry:
            raise KeyError(
                f"No improver registered for task_type={task_type.value}, "
                f"data_type={data_type.value}"
            )
        return self._improver_registry[key]

    def create_updater(self, task_type: TaskType, data_type: DataType):
        """
        Create an instance of the appropriate updater.

        Args:
            task_type: The task type
            data_type: The data type

        Returns:
            An instance of the appropriate updater class
        """
        updater_class = self.get_updater_class(task_type, data_type)
        return updater_class()

    def create_improver(self, task_type: TaskType, data_type: DataType, llm_model):
        """
        Create an instance of the appropriate improver.

        Args:
            task_type: The task type
            data_type: The data type
            llm_model: The LLM model instance to use

        Returns:
            An instance of the appropriate improver class
        """
        improver_class = self.get_improver_class(task_type, data_type)
        return improver_class(llm_model)


def create_default_registry() -> ModelTypeRegistry:
    """
    Create and populate the default model type registry.

    Returns:
        A ModelTypeRegistry instance with all standard components registered
    """
    from src.dynamic_models.dynamic_model_updater import (
        DynamicModelUpdater,
        DynamicRegressionModelUpdater,
        DynamicImageModelUpdater,
        DynamicImageRegressionModelUpdater,
    )
    from src.core.llm_improver import (
        NNLLMImprover,
        NNRegressionLLMImprover,
        NNImageLLMImprover,
        NNImageRegressionLLMImprover,
    )

    registry = ModelTypeRegistry()

    registry.register_updater(
        TaskType.CLASSIFICATION, DataType.TABULAR, DynamicModelUpdater
    )
    registry.register_updater(
        TaskType.REGRESSION, DataType.TABULAR, DynamicRegressionModelUpdater
    )
    registry.register_updater(
        TaskType.CLASSIFICATION, DataType.IMAGE, DynamicImageModelUpdater
    )
    registry.register_updater(
        TaskType.REGRESSION, DataType.IMAGE, DynamicImageRegressionModelUpdater
    )

    registry.register_improver(TaskType.CLASSIFICATION, DataType.TABULAR, NNLLMImprover)
    registry.register_improver(
        TaskType.REGRESSION, DataType.TABULAR, NNRegressionLLMImprover
    )
    registry.register_improver(
        TaskType.CLASSIFICATION, DataType.IMAGE, NNImageLLMImprover
    )
    registry.register_improver(
        TaskType.REGRESSION, DataType.IMAGE, NNImageRegressionLLMImprover
    )

    return registry
