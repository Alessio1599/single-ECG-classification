"""
Here I can save all the function that I will use to preprocess the data.

then in the main file I can import this file and use the functions that I need.

Example usage:
pipeline = Pipeline([
    proc.read_raw_bids_root,
    proc.crop,
    proc.filter,
    proc.resample,
    proc.make_epochs,
    proc.select_channels,
    proc.set_eeg_reference,
    proc.save_epochs,
    proc.compute_meeglet_features,
    proc.save_features
])

Can be useful in case we want to define different pipelines
"""

from pathlib import Path
from types import SimpleNamespace


""" In this way I can use for both train and test
def preprocess_data(input_path, output_path):
    # Load raw data
    raw_data = load_raw_data(input_path)
    
    # Apply transformations (e.g., normalization, filtering, etc.)
    processed_data = transform_data(raw_data)
    
    # Save processed data
    save_data(processed_data, output_path)
"""
def make_config(
    raw_root: str | Path,
    deriv_root: str | Path,
):
    return SimpleNamespace(
        raw_root=Path(raw_root),
        deriv_root=Path(deriv_root),
    )


def split_feature_label(data):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return x, y

