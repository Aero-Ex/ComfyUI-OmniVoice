import os

import folder_paths

from .nodes_omnivoice import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

folder_paths.add_model_folder_path("omnivoice", os.path.join(folder_paths.models_dir, "omnivoice"))
