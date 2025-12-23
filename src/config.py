# config.py

import os
import yaml
from transformers import EsmConfig


def load_config(config_path: str | None = None):
    """
    Load YAML configuration.

    By default, loads `config.yaml` that lives alongside this file (src/config.yaml),
    regardless of the current working directory.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load the YAML configuration once and cache it. Can be overridden via set_config().
_config = load_config()


def set_config(config_path: str):
    """
    Reload and set the global configuration from a specific path.
    """
    global _config
    _config = load_config(config_path)


def get_model_config() -> EsmConfig:
    """
    Returns a Hugging Face EsmConfig object built from the model parameters
    in the YAML config.

    The YAML `model` section can include both standard EsmConfig fields
    (e.g., hidden_size, num_hidden_layers, etc.) and extra fields like:

        model_name
        max_batch_size
        hidden_dropout_prob
        attention_probs_dropout_prob

    EsmConfig.from_dict will attach unknown keys as attributes, so they can be
    accessed as `config.model_name`, `config.max_batch_size`, etc.

    This is used in:
      - data_processing.py (max_position_embeddings, max_batch_size)
      - model.py (model_name, dropout overrides, etc.)
    """
    model_cfg_dict = _config["model"]
    return EsmConfig.from_dict(model_cfg_dict)


def get_training_config():
    """
    Returns the training configuration as a dictionary from the YAML config.
    Used in train.py and by the trainer.
    """
    return _config["training"]


def get_data_config():
    """
    Returns the data processing configuration as a dictionary from the YAML config.
    Used in data_processing.py for defaults such as chunk size and directories.
    """
    return _config["data"]