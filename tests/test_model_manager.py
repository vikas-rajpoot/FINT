"""
Unit tests for ModelManager and ModelRegistry.
"""

import asyncio
import json
import pickle
import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch, mock_open

from multi_agent_trading.services.model_manager import (
    ModelManager, ModelRegistry, ModelMetadata, ModelLoadError, ModelValidationError
)


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_model():
    """Create a sample model object for testing."""
    return {"type": "test_model", "parameters": [1, 2, 3], "version": "1.0"}


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        model_id="test_model_123",
        version="v1.0",
        agent_type="MARKET_ANALYST",
        created_at=datetime.utcnow(),
        performance_metrics={"accuracy": 0.85, "precision": 0.82},
        model_path="/path/to/model.pkl",
        checksum="abc123def456",
        size_bytes=1024,
        framework="pytorch",
        description="Test model",
        tags=["test", "v1"]
    )


class TestModelMetadata:
    """Test ModelMetadata class."""
    
    def test_initialization(self, sample_metadata):
        """Test metadata initialization."""
        assert sample_metadata.model_id == "test_model_123"
        assert sample_metadata.version == "v1.0"
        assert sample_metadata.agent_type == "MARKET_ANALYST"
        assert sample_metadata.framework == "pytorch"
        assert sample_metadata.tags == ["test", "v1"]
    
    def test_default_tags(self):
        """Test default tags initialization."""
        metadata = ModelMetadata(
            model_id="test",
            version="v1.0",
            agent_type="TEST",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path="/test",
            checksum="test",
            size_bytes=100
        )
        
        assert metadata.tags == []
    
    def test_serialization(self, sample_metadata):
        """Test to_dict and from_dict methods."""
        data = sample_metadata.to_dict()
        
        assert data["model_id"] == sample_metadata.model_id
        assert data["version"] == sample_metadata.version
        assert data["agent_type"] == sample_metadata.agent_type
        assert isinstance(data["created_at"], str)  # Should be ISO format
        
        # Test deserialization
        restored = ModelMetadata.from_dict(data)
        
        assert restored.model_id == sample_metadata.model_id
        assert restored.version == sample_metadata.version
        assert restored.agent_type == sample_metadata.agent_type
        assert restored.created_at == sample_metadata.created_at


class TestModelRegistry:
    """Test ModelRegistry class."""
    
    @pytest.fixture
    def registry(self, temp_model_dir):
        """Create model registry with temporary directory."""
        return ModelRegistry(str(Path(temp_model_dir) / "registry"))
    
    @pytest.mark.asyncio
    async def test_initialization(self, registry):
        """Test registry initialization."""
        assert registry.registry_path.exists()
        assert registry._models == {}
    
    @pytest.mark.asyncio
    async def test_register_model(self, registry, sample_metadata):
        """Test model registration."""
        await registry.register_model(sample_metadata)
        
        model_key = f"{sample_metadata.agent_type}:{sample_metadata.version}"
        assert model_key in registry._models
        assert registry._models[model_key] == sample_metadata
    
    @pytest.mark.asyncio
    async def test_get_model(self, registry, sample_metadata):
        """Test getting model by agent type and version."""
        await registry.register_model(sample_metadata)
        
        retrieved = registry.get_model(sample_metadata.agent_type, sample_metadata.version)
        assert retrieved == sample_metadata
        
        # Test non-existent model
        not_found = registry.get_model("NONEXISTENT", "v1.0")
        assert not_found is None
    
    @pytest.mark.asyncio
    async def test_list_models(self, registry, sample_metadata):
        """Test listing models."""
        # Register multiple models
        metadata1 = sample_metadata
        metadata2 = ModelMetadata(
            model_id="test_model_456",
            version="v2.0",
            agent_type="RISK_MANAGER",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path="/path2",
            checksum="def456",
            size_bytes=2048
        )
        
        await registry.register_model(metadata1)
        await registry.register_model(metadata2)
        
        # Test listing all models
        all_models = registry.list_models()
        assert len(all_models) == 2
        
        # Test filtering by agent type
        analyst_models = registry.list_models("MARKET_ANALYST")
        assert len(analyst_models) == 1
        assert analyst_models[0] == metadata1
    
    @pytest.mark.asyncio
    async def test_get_latest_version(self, registry):
        """Test getting latest version of a model."""
        # Create models with different timestamps
        older_metadata = ModelMetadata(
            model_id="old_model",
            version="v1.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime(2023, 1, 1),
            performance_metrics={},
            model_path="/old",
            checksum="old",
            size_bytes=100
        )
        
        newer_metadata = ModelMetadata(
            model_id="new_model",
            version="v2.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime(2023, 12, 31),
            performance_metrics={},
            model_path="/new",
            checksum="new",
            size_bytes=200
        )
        
        await registry.register_model(older_metadata)
        await registry.register_model(newer_metadata)
        
        latest = registry.get_latest_version("MARKET_ANALYST")
        assert latest == newer_metadata
        
        # Test non-existent agent type
        not_found = registry.get_latest_version("NONEXISTENT")
        assert not_found is None
    
    @pytest.mark.asyncio
    async def test_save_and_load_registry(self, registry, sample_metadata):
        """Test saving and loading registry from disk."""
        # Register a model
        await registry.register_model(sample_metadata)
        
        # Create new registry instance with same path
        new_registry = ModelRegistry(str(registry.registry_path))
        await new_registry.load_registry()
        
        # Should have loaded the model
        model_key = f"{sample_metadata.agent_type}:{sample_metadata.version}"
        assert model_key in new_registry._models
        assert new_registry._models[model_key].model_id == sample_metadata.model_id


class TestModelManager:
    """Test ModelManager class."""
    
    @pytest.fixture
    def model_manager(self, temp_model_dir):
        """Create model manager with temporary directory."""
        return ModelManager("test_agent", "v1.0", temp_model_dir)
    
    @pytest.mark.asyncio
    async def test_initialization(self, model_manager):
        """Test model manager initialization."""
        await model_manager.initialize()
        
        assert model_manager.agent_id == "test_agent"
        assert model_manager.model_version == "v1.0"
        assert model_manager.model_path.exists()
        assert model_manager._current_model is None
        assert model_manager._current_metadata is None
    
    @pytest.mark.asyncio
    async def test_save_model(self, model_manager, sample_model, temp_model_dir):
        """Test saving model to disk."""
        await model_manager.initialize()
        
        metadata = ModelMetadata(
            model_id="test_save_model",
            version="v1.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime.utcnow(),
            performance_metrics={"accuracy": 0.9},
            model_path="",  # Will be set by save_model
            checksum="",    # Will be calculated
            size_bytes=0    # Will be calculated
        )
        
        success = await model_manager.save_model(sample_model, metadata)
        assert success is True
        
        # Check that file was created
        model_file = Path(metadata.model_path)
        assert model_file.exists()
        
        # Check that metadata was updated
        assert metadata.checksum != ""
        assert metadata.size_bytes > 0
    
    @pytest.mark.asyncio
    async def test_load_from_path(self, model_manager, sample_model, temp_model_dir):
        """Test loading model from file path."""
        await model_manager.initialize()
        
        # Create a temporary model file
        model_file = Path(temp_model_dir) / "test_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(sample_model, f)
        
        success = await model_manager.load_model(str(model_file))
        assert success is True
        assert model_manager._current_model == sample_model
        assert model_manager._current_metadata is not None
    
    @pytest.mark.asyncio
    async def test_load_from_registry(self, model_manager, sample_model, temp_model_dir):
        """Test loading model from registry."""
        await model_manager.initialize()
        
        # First save a model to registry
        metadata = ModelMetadata(
            model_id="registry_test",
            version="v1.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path="",
            checksum="",
            size_bytes=0
        )
        
        await model_manager.save_model(sample_model, metadata)
        
        # Clear current model
        model_manager._current_model = None
        model_manager._current_metadata = None
        
        # Load from registry
        success = await model_manager.load_model(agent_type="MARKET_ANALYST")
        assert success is True
        assert model_manager._current_model == sample_model
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self, model_manager):
        """Test loading from non-existent file."""
        await model_manager.initialize()
        
        success = await model_manager.load_model("/nonexistent/path.pkl")
        assert success is False
        assert model_manager._current_model is None
    
    @pytest.mark.asyncio
    async def test_update_model(self, model_manager, sample_model, temp_model_dir):
        """Test updating to new model version."""
        await model_manager.initialize()
        
        # Save initial model
        metadata_v1 = ModelMetadata(
            model_id="update_test_v1",
            version="v1.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path="",
            checksum="",
            size_bytes=0
        )
        
        await model_manager.save_model(sample_model, metadata_v1)
        await model_manager.load_model(agent_type="MARKET_ANALYST")
        
        # Save new version
        new_model = {"type": "updated_model", "version": "2.0"}
        metadata_v2 = ModelMetadata(
            model_id="update_test_v2",
            version="v2.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path="",
            checksum="",
            size_bytes=0
        )
        
        await model_manager.save_model(new_model, metadata_v2)
        
        # Update to new version
        success = await model_manager.update_model("v2.0", "MARKET_ANALYST")
        assert success is True
        assert model_manager.model_version == "v2.0"
        assert model_manager._current_model == new_model
        assert model_manager._backup_model == sample_model  # Should backup old model
    
    @pytest.mark.asyncio
    async def test_rollback_model(self, model_manager, sample_model, temp_model_dir):
        """Test rolling back to previous model version."""
        await model_manager.initialize()
        
        # Set up models for rollback test
        old_model = sample_model
        new_model = {"type": "new_model", "version": "2.0"}
        
        # Simulate having current and backup models
        model_manager._current_model = new_model
        model_manager._current_metadata = ModelMetadata(
            model_id="new", version="v2.0", agent_type="TEST",
            created_at=datetime.utcnow(), performance_metrics={},
            model_path="", checksum="", size_bytes=0
        )
        model_manager._backup_model = old_model
        model_manager._backup_metadata = ModelMetadata(
            model_id="old", version="v1.0", agent_type="TEST",
            created_at=datetime.utcnow(), performance_metrics={},
            model_path="", checksum="", size_bytes=0
        )
        model_manager.model_version = "v2.0"
        
        # Perform rollback
        success = await model_manager.rollback_model()
        assert success is True
        assert model_manager._current_model == old_model
        assert model_manager.model_version == "v1.0"
        assert model_manager._backup_model == new_model  # Should swap
    
    @pytest.mark.asyncio
    async def test_rollback_without_backup(self, model_manager):
        """Test rollback when no backup is available."""
        await model_manager.initialize()
        
        success = await model_manager.rollback_model()
        assert success is False
    
    def test_get_model(self, model_manager, sample_model):
        """Test getting current model."""
        model_manager._current_model = sample_model
        
        retrieved = model_manager.get_model()
        assert retrieved == sample_model
        
        # Test when no model is loaded
        model_manager._current_model = None
        retrieved = model_manager.get_model()
        assert retrieved is None
    
    def test_get_model_info(self, model_manager, sample_metadata):
        """Test getting model information."""
        model_manager._current_model = {"test": "model"}
        model_manager._current_metadata = sample_metadata
        model_manager._backup_model = {"backup": "model"}
        
        info = model_manager.get_model_info()
        
        assert info["agent_id"] == "test_agent"
        assert info["model_version"] == "v1.0"
        assert info["loaded"] is True
        assert info["has_backup"] is True
        assert info["model_id"] == sample_metadata.model_id
        assert "created_at" in info
        assert "performance_metrics" in info
    
    @pytest.mark.asyncio
    async def test_validate_model(self, model_manager, sample_model, temp_model_dir):
        """Test model validation."""
        await model_manager.initialize()
        
        # Save and load a model
        metadata = ModelMetadata(
            model_id="validation_test",
            version="v1.0",
            agent_type="MARKET_ANALYST",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path="",
            checksum="",
            size_bytes=0
        )
        
        await model_manager.save_model(sample_model, metadata)
        await model_manager.load_model(agent_type="MARKET_ANALYST")
        
        # Validation should pass
        is_valid = await model_manager.validate_model()
        assert is_valid is True
        
        # Test validation failure when file is missing
        os.remove(model_manager._current_metadata.model_path)
        is_valid = await model_manager.validate_model()
        assert is_valid is False
    
    def test_record_performance(self, model_manager):
        """Test recording performance metrics."""
        metrics = {"accuracy": 0.95, "precision": 0.92, "recall": 0.88}
        
        model_manager.record_performance(metrics)
        
        history = model_manager.get_performance_history()
        assert len(history) == 1
        assert history[0]["metrics"] == metrics
        assert history[0]["model_version"] == "v1.0"
        assert "timestamp" in history[0]
    
    def test_performance_history_limit(self, model_manager):
        """Test performance history size limit."""
        # Record more than 100 metrics
        for i in range(105):
            model_manager.record_performance({"metric": i})
        
        history = model_manager.get_performance_history()
        assert len(history) <= 100
        
        # Should keep the most recent ones
        assert history[-1]["metrics"]["metric"] == 104
    
    @pytest.mark.asyncio
    async def test_cleanup(self, model_manager, sample_model):
        """Test cleanup functionality."""
        await model_manager.initialize()
        
        # Set up some state
        model_manager._current_model = sample_model
        model_manager._backup_model = sample_model
        model_manager.record_performance({"test": 1})
        
        await model_manager.cleanup()
        
        # Should clear all state
        assert model_manager._current_model is None
        assert model_manager._current_metadata is None
        assert model_manager._backup_model is None
        assert model_manager._backup_metadata is None
        assert len(model_manager._performance_history) == 0


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_model_load_error(self, temp_model_dir):
        """Test ModelLoadError handling."""
        manager = ModelManager("test", "v1.0", temp_model_dir)
        await manager.initialize()
        
        # Try to load from registry without agent type
        success = await manager.load_model()
        assert success is False
    
    @pytest.mark.asyncio
    async def test_model_validation_error(self, temp_model_dir, sample_model):
        """Test ModelValidationError handling."""
        manager = ModelManager("test", "v1.0", temp_model_dir)
        await manager.initialize()
        
        # Create a model file
        model_file = Path(temp_model_dir) / "test.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(sample_model, f)
        
        # Create metadata with wrong checksum
        metadata = ModelMetadata(
            model_id="test",
            version="v1.0",
            agent_type="TEST",
            created_at=datetime.utcnow(),
            performance_metrics={},
            model_path=str(model_file),
            checksum="wrong_checksum",
            size_bytes=100
        )
        
        # This should fail validation
        success = await manager._load_from_metadata(metadata)
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])