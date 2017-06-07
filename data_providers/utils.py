from .data import DataProvider

def get_data_provider_by_name(name, train_params):
    """Return required data provider class"""
    if name == 'UCF101':
        return DataProvider(**train_params)
    if name == 'MERL':
        return DataProvider(**train_params)
    if name == 'KTH':
        return DataProvider(**train_params)
    else:
        print("Sorry, data provider for `%s` dataset "
              "was not implemented yet" % name)
        exit()
