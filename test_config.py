"""
Test script for configuration system.
"""
import os
import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.absolute()))

from src.config import Config, get_default_config

def test_config_creation():
    """Test creating and saving a config."""
    print("Testing config creation...")
    
    # Create default config
    config = get_default_config()
    
    # Print some config values
    print(f"Project: {config.project_name}")
    print(f"Model: {config.model.num_res_blocks} res blocks, {config.model.num_filters} filters")
    print(f"Training: {config.training.num_epochs} epochs, lr={config.training.learning_rate}")
    
    # Test saving and loading
    test_path = "test_config.json"
    config.save(test_path)
    print(f"Saved config to {test_path}")
    
    loaded_config = Config.load(test_path)
    print(f"Loaded config matches original: {config.to_dict() == loaded_config.to_dict()}")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("Config test completed successfully!")

def test_default_config_file():
    """Test loading the default config file."""
    print("\nTesting default config file...")
    
    config_path = "configs/default_config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        return
    
    config = Config.load(config_path)
    print(f"Loaded config from {config_path}")
    print(f"Project: {config.project_name}")
    print(f"Model: {config.model.num_res_blocks} res blocks, {config.model.num_filters} filters")
    print(f"Training: {config.training.num_epochs} epochs, lr={config.training.learning_rate}")
    
    # Test saving the loaded config
    test_path = "test_loaded_config.json"
    config.save(test_path)
    print(f"Saved loaded config to {test_path}")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("Default config file test completed!")

if __name__ == "__main__":
    test_config_creation()
    test_default_config_file()
