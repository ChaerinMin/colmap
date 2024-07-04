import logging
import os
import shutil


def create_dir(path: str):
    """Creates a directory also deleting previous one if it exissts"""
    logging.warning(f"Deleting files at {path}")
    try:
        shutil.rmtree(path)
    except:
        pass
    os.makedirs(path)