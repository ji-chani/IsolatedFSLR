# module for creating paths to generate the FSL dataset
import numpy as np
import os

class FSL105:
    def __init__(self, folder_path:str, file_extension:str='MOV'):
        self.folder_path = folder_path
        self.extension = file_extension

    def load_data(self, selected_classes:int | list) -> dict:
        videos_per_class = get_files_per_class(self.folder_path)                    # dict: key=class, value=file_paths
        videos_per_class_subset = subset_data(videos_per_class, selected_classes)   # dict: key=class, value=file_paths
        return reconstruct_data(videos_per_class_subset)                            # dict: key=['target','path'], value=[class,file_paths]

############ HELPER FUNCTIONS #################
def get_files_per_class(folder_path:str) -> dict:
    """
    Creates a dictionary with keys = class and values = file paths. Keys sorted by class index.
    """
    videos_per_class = {}
    for cls in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, cls)
        videos_per_class[int(cls)] = [os.path.join(subfolder_path,f) for f in os.listdir(subfolder_path) if f.endswith("MOV")]
    return {cls: videos_per_class[cls] for cls in sorted(videos_per_class)}

def subset_data(videos_per_class:dict, selected_classes:int|list):
    """
    Obtain a subset of the complete dataset (videos_per_class).
    If `type(selected_classes) = int` -> first `selected_classes` are obtained.
    If `type(selected_classes) = list` -> classes in `selected_classes` are obtained.
    """
    class_subset = np.arange(selected_classes) if type(selected_classes) == int else np.array(sorted(selected_classes))
    return {cls: videos_per_class[cls] for cls in class_subset}

def reconstruct_data(videos_per_class:dict):
    """
    Transforms data into dictionary: keys = ['path', 'target'], values = [file_paths, class]
    """
    dataset = {'target': [], 'path': []}
    for cls in videos_per_class.keys():
        [[dataset['path'].append(f) for f in videos_per_class[cls]]]
        [[dataset['target'].append(int(cls)) for _ in videos_per_class[cls]]]
    return dataset