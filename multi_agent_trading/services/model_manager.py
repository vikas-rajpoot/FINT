"""
Model management service for ML model versioning and deployment.
"""

import asyncio
import json
import logging
import os
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import aiofiles


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class ModelValidationError(Exception):
    """Exception raised when model validation fails."""
    pass


@dataclass
class ModelMetadata:
    """Metadata for ML models."""
    model_id: str
    version: str
    agent_type: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    model_path: str
    checksum: str
    size_bytes: int
    framework: str = "pytorch"
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create instance from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class ModelRegistry:
    """Registry for managing model metadata and versions."""
    
    def __init__(self, registry_path: str = ".models/registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store registry metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "models.json"
        self.logger = logging.getLogger(self.__class__.__name__)
        self._models: Dict[str, ModelMetadata] = {}
    
    async def load_registry(self) -> None:
        """Load model registry from disk."""
        try:
            if self.metadata_file.exists():
                async with aiofiles.open(self.metadata_file, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    for model_key, model_data in data.items():
                        self._models[model_key] = ModelMetadata.from_dict(model_data)
                        
                self.logger.info(f"Loaded {len(self._models)} models from registry")
            else:
                self.logger.info("No existing registry found, starting fresh")
                
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
            self._models = {}
    
    async def save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            data = {key: model.to_dict() for key, model in self._models.items()}
            
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
            self.logger.debug("Registry saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    async def register_model(self, metadata: ModelMetadata) -> None:
        """
        Register a new model version.
        
        Args:
            metadata: Model metadata to register
        """
        model_key = f"{metadata.agent_type}:{metadata.version}"
        self._models[model_key] = metadata
        await self.save_registry()
        self.logger.info(f"Registered model {model_key}")
    
    def get_model(self, agent_type: str, version: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by agent type and version.
        
        Args:
            agent_type: Type of agent
            version: Model version
            
        Returns:
            Model metadata if found, None otherwise
        """
        model_key = f"{agent_type}:{version}"
        return self._models.get(model_key)
    
    def list_models(self, agent_type: Optional[str] = None) -> List[ModelMetadata]:
        """
        List all models, optionally filtered by agent type.
        
        Args:
            agent_type: Optional agent type filter
            
        Returns:
            List of model metadata
        """
        if agent_type:
            return [model for key, model in self._models.items() 
                   if model.agent_type == agent_type]
        return list(self._models.values())
    
    def get_latest_version(self, agent_type: str) -> Optional[ModelMetadata]:
        """
        Get the latest version of a model for an agent type.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            Latest model metadata if found, None otherwise
        """
        models = self.list_models(agent_type)
        if not models:
            return None
        
        # Sort by creation date and return the latest
        return max(models, key=lambda m: m.created_at)


class ModelManager:
    """
    Manages ML models for agents including versioning and deployment.
    """
    
    def __init__(self, agent_id: str, model_version: str, model_path: str = ".models"):
        """
        Initialize model manager.
        
        Args:
            agent_id: ID of the agent using this manager
            model_version: Version of the model to load
            model_path: Base path for model storage
        """
        self.agent_id = agent_id
        self.model_version = model_version
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")
        
        # Model state
        self._current_model = None
        self._current_metadata: Optional[ModelMetadata] = None
        self._backup_model = None
        self._backup_metadata: Optional[ModelMetadata] = None
        
        # Registry
        self.registry = ModelRegistry(str(self.model_path / "registry"))
        
        # Performance tracking
        self._performance_history: List[Dict[str, Any]] = []
        
    async def initialize(self) -> None:
        """Initialize the model manager and load registry."""
        await self.registry.load_registry()
        self.logger.info(f"Model manager initialized for agent {self.agent_id}")
    
    async def load_model(self, model_path: Optional[str] = None, 
                        agent_type: Optional[str] = None) -> bool:
        """
        Load model from registry or path.
        
        Args:
            model_path: Optional path to model file
            agent_type: Optional agent type for registry lookup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path:
                # Load from specific path
                return await self._load_from_path(model_path)
            else:
                # Load from registry
                if not agent_type:
                    raise ModelLoadError("Agent type required for registry lookup")
                
                metadata = self.registry.get_model(agent_type, self.model_version)
                if not metadata:
                    # Try to get latest version
                    metadata = self.registry.get_latest_version(agent_type)
                    if not metadata:
                        raise ModelLoadError(f"No model found for {agent_type}:{self.model_version}")
                    
                    self.logger.warning(f"Version {self.model_version} not found, using latest: {metadata.version}")
                    self.model_version = metadata.version
                
                return await self._load_from_metadata(metadata)
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    async def _load_from_path(self, model_path: str) -> bool:
        """Load model from file path."""
        try:
            path = Path(model_path)
            if not path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")
            
            # Calculate checksum
            checksum = await self._calculate_checksum(path)
            
            # Load model
            async with aiofiles.open(path, 'rb') as f:
                model_data = await f.read()
                self._current_model = pickle.loads(model_data)
            
            # Create metadata
            self._current_metadata = ModelMetadata(
                model_id=f"{self.agent_id}_{int(datetime.utcnow().timestamp())}",
                version=self.model_version,
                agent_type=self.agent_id.split('_')[0] if '_' in self.agent_id else "unknown",
                created_at=datetime.utcnow(),
                performance_metrics={},
                model_path=str(path),
                checksum=checksum,
                size_bytes=path.stat().st_size,
                description=f"Model loaded from {model_path}"
            )
            
            self.logger.info(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from path {model_path}: {e}")
            return False
    
    async def _load_from_metadata(self, metadata: ModelMetadata) -> bool:
        """Load model using metadata from registry."""
        try:
            # Verify file exists and checksum matches
            model_path = Path(metadata.model_path)
            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {metadata.model_path}")
            
            # Verify checksum
            current_checksum = await self._calculate_checksum(model_path)
            if current_checksum != metadata.checksum:
                raise ModelValidationError(f"Model checksum mismatch for {metadata.model_id}")
            
            # Backup current model if exists
            if self._current_model is not None:
                self._backup_model = self._current_model
                self._backup_metadata = self._current_metadata
            
            # Load new model
            async with aiofiles.open(model_path, 'rb') as f:
                model_data = await f.read()
                self._current_model = pickle.loads(model_data)
            
            self._current_metadata = metadata
            
            self.logger.info(f"Successfully loaded model {metadata.model_id} version {metadata.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from metadata: {e}")
            return False
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def save_model(self, model: Any, metadata: ModelMetadata) -> bool:
        """
        Save model to disk and register in registry.
        
        Args:
            model: Model object to save
            metadata: Model metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create model directory
            model_dir = self.model_path / metadata.agent_type / metadata.version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_file = model_dir / f"{metadata.model_id}.pkl"
            async with aiofiles.open(model_file, 'wb') as f:
                await f.write(pickle.dumps(model))
            
            # Update metadata with file path and checksum
            metadata.model_path = str(model_file)
            metadata.checksum = await self._calculate_checksum(model_file)
            metadata.size_bytes = model_file.stat().st_size
            
            # Register in registry
            await self.registry.register_model(metadata)
            
            self.logger.info(f"Successfully saved model {metadata.model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    async def update_model(self, new_version: str, agent_type: Optional[str] = None) -> bool:
        """
        Update to new model version with rollback capability.
        
        Args:
            new_version: New model version to load
            agent_type: Optional agent type for registry lookup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            old_version = self.model_version
            self.logger.info(f"Updating model from {old_version} to {new_version}")
            
            # Load new model
            self.model_version = new_version
            success = await self.load_model(agent_type=agent_type)
            
            if success:
                self.logger.info(f"Successfully updated to model version {new_version}")
                return True
            else:
                # Rollback on failure
                self.model_version = old_version
                if self._backup_model is not None:
                    self._current_model = self._backup_model
                    self._current_metadata = self._backup_metadata
                    self._backup_model = None
                    self._backup_metadata = None
                
                self.logger.error(f"Failed to update model, rolled back to {old_version}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False
    
    async def rollback_model(self) -> bool:
        """
        Rollback to previous model version.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._backup_model is None or self._backup_metadata is None:
                self.logger.warning("No backup model available for rollback")
                return False
            
            # Swap current and backup
            current_model = self._current_model
            current_metadata = self._current_metadata
            
            self._current_model = self._backup_model
            self._current_metadata = self._backup_metadata
            
            self._backup_model = current_model
            self._backup_metadata = current_metadata
            
            if self._current_metadata:
                self.model_version = self._current_metadata.version
            
            self.logger.info(f"Successfully rolled back to model version {self.model_version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return False
    
    def get_model(self) -> Any:
        """
        Get current loaded model.
        
        Returns:
            Current model object or None if not loaded
        """
        return self._current_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about current model.
        
        Returns:
            Model information dictionary
        """
        info = {
            "agent_id": self.agent_id,
            "model_version": self.model_version,
            "loaded": self._current_model is not None,
            "has_backup": self._backup_model is not None
        }
        
        if self._current_metadata:
            info.update({
                "model_id": self._current_metadata.model_id,
                "created_at": self._current_metadata.created_at.isoformat(),
                "performance_metrics": self._current_metadata.performance_metrics,
                "size_bytes": self._current_metadata.size_bytes,
                "framework": self._current_metadata.framework
            })
        
        return info
    
    async def validate_model(self) -> bool:
        """
        Validate current model integrity.
        
        Returns:
            True if model is valid, False otherwise
        """
        try:
            if self._current_model is None or self._current_metadata is None:
                return False
            
            # Check if file still exists
            model_path = Path(self._current_metadata.model_path)
            if not model_path.exists():
                self.logger.error(f"Model file missing: {self._current_metadata.model_path}")
                return False
            
            # Verify checksum
            current_checksum = await self._calculate_checksum(model_path)
            if current_checksum != self._current_metadata.checksum:
                self.logger.error("Model checksum validation failed")
                return False
            
            self.logger.debug("Model validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return False
    
    def record_performance(self, metrics: Dict[str, float]) -> None:
        """
        Record performance metrics for the current model.
        
        Args:
            metrics: Performance metrics to record
        """
        performance_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": self.model_version,
            "metrics": metrics
        }
        
        self._performance_history.append(performance_record)
        
        # Keep only last 100 records
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-100:]
        
        self.logger.debug(f"Recorded performance metrics: {metrics}")
    
    def get_performance_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get performance history for the model.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of performance records
        """
        return self._performance_history[-limit:]
    
    async def cleanup(self) -> None:
        """Cleanup model resources."""
        try:
            self.logger.info(f"Cleaning up model resources for agent {self.agent_id}")
            
            # Save registry before cleanup
            await self.registry.save_registry()
            
            # Clear model references
            self._current_model = None
            self._current_metadata = None
            self._backup_model = None
            self._backup_metadata = None
            self._performance_history.clear()
            
            self.logger.info("Model cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")