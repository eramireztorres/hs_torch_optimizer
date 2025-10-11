"""Tests for ModelTypeRegistry."""

import pytest
from src.core.model_type_registry import (
    ModelTypeRegistry,
    TaskType,
    DataType,
    create_default_registry,
)


@pytest.mark.unit
class TestModelTypeRegistry:
    """Test suite for ModelTypeRegistry."""

    def test_init(self):
        """Test initialization."""
        registry = ModelTypeRegistry()
        assert registry._updater_registry == {}
        assert registry._improver_registry == {}

    def test_register_updater(self):
        """Test registering an updater."""
        registry = ModelTypeRegistry()

        class TestUpdater:
            pass

        registry.register_updater(
            TaskType.CLASSIFICATION, DataType.TABULAR, TestUpdater
        )

        updater_class = registry.get_updater_class(
            TaskType.CLASSIFICATION, DataType.TABULAR
        )
        assert updater_class == TestUpdater

    def test_register_improver(self):
        """Test registering an improver."""
        registry = ModelTypeRegistry()

        class TestImprover:
            pass

        registry.register_improver(TaskType.REGRESSION, DataType.IMAGE, TestImprover)

        improver_class = registry.get_improver_class(
            TaskType.REGRESSION, DataType.IMAGE
        )
        assert improver_class == TestImprover

    def test_get_updater_class_not_registered(self):
        """Test getting unregistered updater raises error."""
        registry = ModelTypeRegistry()

        with pytest.raises(KeyError, match="No updater registered"):
            registry.get_updater_class(TaskType.CLASSIFICATION, DataType.TABULAR)

    def test_get_improver_class_not_registered(self):
        """Test getting unregistered improver raises error."""
        registry = ModelTypeRegistry()

        with pytest.raises(KeyError, match="No improver registered"):
            registry.get_improver_class(TaskType.CLASSIFICATION, DataType.TABULAR)

    def test_create_updater(self):
        """Test creating updater instance."""
        registry = ModelTypeRegistry()

        class TestUpdater:
            def __init__(self):
                self.name = "test"

        registry.register_updater(
            TaskType.CLASSIFICATION, DataType.TABULAR, TestUpdater
        )

        updater = registry.create_updater(TaskType.CLASSIFICATION, DataType.TABULAR)
        assert isinstance(updater, TestUpdater)
        assert updater.name == "test"

    def test_create_improver(self):
        """Test creating improver instance."""
        registry = ModelTypeRegistry()

        class TestImprover:
            def __init__(self, llm_model):
                self.llm = llm_model

        registry.register_improver(TaskType.REGRESSION, DataType.IMAGE, TestImprover)

        mock_llm = object()
        improver = registry.create_improver(
            TaskType.REGRESSION, DataType.IMAGE, mock_llm
        )
        assert isinstance(improver, TestImprover)
        assert improver.llm == mock_llm

    def test_create_default_registry(self):
        """Test creating default registry."""
        registry = create_default_registry()

        # Check all combinations are registered
        for task_type in [TaskType.CLASSIFICATION, TaskType.REGRESSION]:
            for data_type in [DataType.TABULAR, DataType.IMAGE]:
                # Should not raise
                registry.get_updater_class(task_type, data_type)
                registry.get_improver_class(task_type, data_type)

    def test_task_type_enum(self):
        """Test TaskType enum."""
        assert TaskType.CLASSIFICATION.value == "classification"
        assert TaskType.REGRESSION.value == "regression"

    def test_data_type_enum(self):
        """Test DataType enum."""
        assert DataType.TABULAR.value == "tabular"
        assert DataType.IMAGE.value == "image"
