from .data import DataProvider

"""Args
    path: path to the video data folder
    """"
def get_data_provider_by_path(path, train_params):
    """Return required data provider class"""
    return DataProvider(path, **train_params)

