import json
import pathlib as pth
from typing import Union, Optional
import torch
import torch.nn as nn
import ast


def wrap_hist(**kwargs) -> dict:
    return kwargs

# Files below are meant to be used in conjunction with the training script and each other. 
# Save model saves it to path (new path if existing_ok is False) and returns the final_path. Final_path should usually be the input when saving new .json/ model.
def convert_str_values(config_dict: dict) -> dict:
    """Convert string representations of lists, tuples, and dictionaries in the config_dict to their actual types."""
    for key, value in config_dict.copy().items():
        if isinstance(value, str):
            if "comment" in key.lower():
                continue

            try:
                config_dict[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # Keep the original string if it cannot be evaluated

    return config_dict

def _existing_files(path: Union[str, pth.Path]) -> int:
    """
    Returns the number of existing files with the same base name as the given file path.
    """

    path = pth.Path(path)
    base_name = path.stem.split('_')[0] # Base name (e.g., 'ResNet')
    parent_dir = path.parent

    # Iterate over all files matching the base pattern and count them
    count = 0
    for f in parent_dir.glob(f'{base_name}_*.pt'):
        count += 1

    return count

def save_model(path: Union[str, pth.Path],
               model: nn.Module,
               existing_ok: bool = True) -> pth.Path:

    """
    Saves PyTorch model state dictionary to a file with automatic version numbering. Creates parent directories if they don't exist. 
    If existing_ok=False, deletes any existing model files with the same base name.
    """

    path = pth.Path(path)
    base_name = path.stem.split('_')[0] # Base name (e.g., 'ResNet')
    parent_dir = path.parent
    
    # Ensure the directory exists
    parent_dir.mkdir(parents=True, exist_ok=True) 

    max_existing_num = _existing_files(path)
    new_num = max_existing_num + 1

    if existing_ok:
        pass
    else:
        # Iterate over all files matching the base pattern and delete them (Pathlib)
        for f in parent_dir.glob(f'{base_name}_*.pt'):
            f.unlink() # Pathlib method for file deletion
    
    final_path = parent_dir / f'{base_name}_{new_num}.pt'

    # Save the model
    torch.save(model.state_dict(), final_path)
        
        
    return final_path

def load_model(file_path: Union[str, pth.Path],
               model: nn.Module,
               device: Optional[torch.device] = None) -> nn.Module:
 
    file_path = pth.Path(file_path)
    if not file_path.exists():
        raise ValueError(f'Path {file_path} does not exist.')

    if device is not None:
        model_state_dict = torch.load(file_path, map_location=device)
    else:
        model_state_dict = torch.load(file_path, map_location=device)
    
    model.load_state_dict(model_state_dict)
    model.to(device)

    return model

def save2json(data: dict, path: Union[str, pth.Path]) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        path (Union[str, pth.Path]): The path to the JSON file.

    Raises:
        ValueError: If the dictionary cannot be saved to the JSON file.
    """

    path = pth.Path(path)
    
    path = pth.Path(path)
    if path.exists():
        path.unlink()

    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)  # indent=4 for human-readable formatting

    except Exception as e:
        raise (f"Error saving dictionary to {path}: {e}")
    
def load_json(path: Union [str, pth.Path]) -> dict:

    """
    Load a dictionary from a JSON file.

    Args:
        path (Union[str, pth.Path]): The path to the JSON file.

    Returns:
        dict: The loaded dictionary.

    Raises:
        ValueError: If the JSON file cannot be loaded.
    
    """
    
    path = pth.Path(path)

    try:
        with open(path, 'r') as f:
            data = json.load(f)

        return data
    except Exception as e:
        raise (f"Error loading dictionary from {path}: {e}")
