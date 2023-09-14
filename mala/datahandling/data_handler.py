"""DataHandler class that loads and scales data."""
import os

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass

from mala.datahandling.data_handler_base import DataHandlerBase
from mala.common.parameters import Parameters

class DataHandler(DataHandlerBase):
    def __new__(cls, params: Parameters):
        from mala.datahandling.data_handler_graph import DataHandlerGraph
        from mala.datahandling.data_handler_mlp import DataHandlerMLP
        data_handler = None

        # Check if we're accessing through base class.
        # If not, we need to return the correct object directly.
        if cls == DataHandler:
            if params.network.nn_type == "se3_transformer":
                data_handler = super(DataHandler, DataHandlerGraph).__new__(DataHandlerGraph)
            else:
                data_handler = super(DataHandler, DataHandlerMLP).__new__(DataHandlerMLP)

            if data_handler is None:
                raise Exception("Unsupported dataset type.")
        else:
            data_handler = super(DataHandler, cls).__new__(cls)

        return data_handler