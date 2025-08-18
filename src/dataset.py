"""Dataset utilities for BeeVision."""
from pathlib import Path

from roboflow import Roboflow


def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int,
                              fmt: str = "yolov8", dest: str = "data") -> Path:
    """Download a dataset from Roboflow.

    Parameters
    ----------
    api_key: str
        Roboflow API key.
    workspace: str
        Workspace name in Roboflow.
    project: str
        Project slug in the workspace.
    version: int
        Dataset version number.
    fmt: str, optional
        Format to download (default "yolov8").
    dest: str, optional
        Destination directory for the dataset (default "data").

    Returns
    -------
    Path
        Path to the downloaded dataset directory.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download(fmt, location=dest)
    return Path(dataset.location)
