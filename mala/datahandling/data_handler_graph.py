"""DataHandler class that loads and scales data."""
import os


try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch
from torch.utils.data import TensorDataset

from mala.common.parallelizer import printout, barrier
from mala.common.parameters import Parameters, DEFAULT_NP_DATA_DTYPE
from mala.datahandling.data_handler_base import DataHandlerBase
from mala.datahandling.data_scaler import DataScaler
from mala.datahandling.snapshot import Snapshot
from mala.datahandling.graph_dataset import GraphDataset


class DataHandlerGraph(DataHandlerBase):
    """
    Loads and scales graph data.

    Parameters
    ----------
    parameters : mala.common.parameters.Parameters
        Parameters used to create the data handling object.

    descriptor_calculator : mala.descriptors.descriptor.Descriptor
        Used to do unit conversion on input data. If None, then one will
        be created by this class.

    target_calculator : mala.targets.target.Target
        Used to do unit conversion on output data. If None, then one will
        be created by this class.

    input_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the input data. If None, then one will be created by
        this class.

    output_data_scaler : mala.datahandling.data_scaler.DataScaler
        Used to scale the output data. If None, then one will be created by
        this class.

    clear_data : bool
        If true (default), the data list will be cleared upon creation of
        the object.
    """

    ##############################
    # Constructors
    ##############################

    def __init__(self, parameters: Parameters, target_calculator=None,
                 descriptor_calculator=None, input_data_scaler=None,
                 output_data_scaler=None, clear_data=True,
                 ):
        super(DataHandlerGraph, self).__init__(parameters,
                                          target_calculator=target_calculator,
                                          descriptor_calculator=
                                          descriptor_calculator)
        self.params = parameters

        # Data will be scaled per user specification.            
        self.input_data_scaler = input_data_scaler
        if self.input_data_scaler is None:
            self.input_data_scaler \
                = DataScaler(self.parameters.input_rescaling_type,
                             use_horovod=self.use_horovod)

        self.output_data_scaler = output_data_scaler
        if self.output_data_scaler is None:
            self.output_data_scaler \
                = DataScaler(self.parameters.output_rescaling_type,
                             use_horovod=self.use_horovod)

        # Actual data points in the different categories.

        # Number of snapshots in these categories.
        self.nr_training_snapshots = 0
        self.nr_test_snapshots = 0
        self.nr_validation_snapshots = 0

        # Arrays and data sets containing the actual data.
        self.training_data_sets = []
        self.validation_data_sets = []
        self.test_data_sets = []

        self.raw_snapshots_train = []
        self.raw_snapshots_test = []
        self.raw_snapshots_validation = []

        # Needed for the fast tensor data sets.
        self.ldos_grid_batch_size = parameters.running.ldos_grid_batch_size
        if clear_data:
            self.clear_data()

    ##############################
    # Public methods
    ##############################

    # Adding/Deleting data
    ########################

    # Temporary
    def add_raw_snapshot(self, input_path, ldos_path, ldos_shape, add_snapshot_as):
        """
        Add a snapshot to the data pipeline.

        Parameters
        ----------
        input_path : string
            File with saved numpy input array.

        ldos_path : string
            Directory containing ldos_npy_directory.

        ldos_shape : tuple
            Shape of the ldos data.

        add_snapshot_as : string
            Must be "tr", "va" or "te", the snapshot will be added to the
            snapshot list as training, validation or testing snapshot,
            respectively.
        """
        snapshot = {
            "input_path": input_path,
            "ldos_path": ldos_path,
            "ldos_shape": ldos_shape
        }
        if add_snapshot_as == "tr":
            self.raw_snapshots_train.append(snapshot)
        elif add_snapshot_as == "va":
            self.raw_snapshots_validation.append(snapshot)
        elif add_snapshot_as == "te":
            self.raw_snapshots_test.append(snapshot)
        else:
            raise Exception("Invalid snapshot type.")

    def clear_data(self):
        """
        Reset the entire data pipeline.

        Useful when doing multiple investigations in the same python file.
        """
        self.training_data_sets = []
        self.validation_data_sets = []
        self.test_data_sets = []
        self.nr_training_snapshots = 0
        self.nr_test_snapshots = 0
        self.nr_validation_snapshots = 0
        super(DataHandlerGraph, self).clear_data()

    # Preparing data
    ######################

    def prepare_data(self, reparametrize_scaler=True):
        """
        Prepare the data to be used in a training process.

        This includes:

            - Checking snapshots for consistency
            - Parametrizing the DataScalers (if desired)
            - Building DataSet objects.

        Parameters
        ----------
        reparametrize_scaler : bool
            If True (default), the DataScalers are parametrized based on the
            training data.

        """
        # During data loading, there is no need to save target data to
        # calculators.
        # Technically, this would be no issue, but due to technical reasons
        # (i.e. float64 to float32 conversion) saving the data this way
        # may create copies in memory.
        self.target_calculator.save_target_data = False

        # Do a consistency check of the snapshots so that we don't run into
        # an error later. If there is an error, check_snapshots() will raise
        # an exception.
        printout("Checking the snapshots and your inputs for consistency.",
                 min_verbosity=1)
        self._check_snapshots()
        printout("Consistency check successful.", min_verbosity=0)

        # If the DataHandler is used for inference, i.e. no training or
        # validation snapshots have been provided,
        # than we can definitely not reparametrize the DataScalers.
        if self.nr_training_snapshots == 0:
            reparametrize_scaler = False
            if self.input_data_scaler.cantransform is False or \
                    self.output_data_scaler.cantransform is False:
                raise Exception("In inference mode, the DataHandler needs "
                                "parametrized DataScalers, "
                                "while you provided unparametrized "
                                "DataScalers.")

        # Parametrize the scalers, if needed.
        if reparametrize_scaler:
            printout("Initializing the data scalers.", min_verbosity=1)
            self.__parametrize_scalers()
            printout("Data scalers initialized.", min_verbosity=0)
        elif self.parameters.use_lazy_loading is False and \
                self.nr_training_snapshots != 0:
            printout("Data scalers already initilized, loading data to RAM.",
                     min_verbosity=0)

        # Build Datasets.
        printout("Build datasets.", min_verbosity=1)
        self.__build_datasets()
        printout("Build dataset: Done.", min_verbosity=0)

        # After the loading is done, target data can safely be saved again.
        self.target_calculator.save_target_data = True

        # Wait until all ranks are finished with data preparation.
        # It is not uncommon that ranks might be asynchronous in their
        # data preparation by a small amount of minutes. If you notice
        # an elongated wait time at this barrier, check that your file system
        # allows for parallel I/O.
        barrier()

    def prepare_for_testing(self):
        """
        Prepare DataHandler for usage within Tester class.

        Ensures that lazily-loaded data sets do not perform unnecessary I/O
        operations. Only needed in Tester class.
        """
        if self.parameters.use_lazy_loading:
            self.test_data_set.return_outputs_directly = True

    # Training  / Testing
    ######################

    def mix_datasets(self):
        """
        For lazily-loaded data sets, the snapshot ordering is (re-)mixed.

        This applies only to the training data set. For the validation and
        test set it does not matter.
        """
        if self.parameters.use_lazy_loading:
            for dset in self.training_data_sets:
                dset.mix_datasets()

    def get_test_input_gradient(self, snapshot_number):
        """
        Get the gradient of the test inputs for an entire snapshot.

        This gradient will be returned as scaled Tensor.
        The reason the gradient is returned (rather then returning the entire
        inputs themselves) is that by slicing a variable, pytorch no longer
        considers it a "leaf" variable and will stop tracking and evaluating
        its gradient. Thus, it is easier to obtain the gradient and then
        slice it.

        Parameters
        ----------
        snapshot_number : int
            Number of the snapshot for which the entire test inputs.

        Returns
        -------
        torch.Tensor
            Tensor holding the gradient.

        """
        # get the snapshot from the snapshot number
        snapshot = self.parameters.snapshot_directories_list[snapshot_number]
        
        if self.parameters.use_lazy_loading:
            # This fails if an incorrect snapshot was loaded.
            if self.test_data_sets[0].currently_loaded_file != snapshot_number:
                raise Exception("Cannot calculate gradients, wrong file "
                                "was lazily loaded.")
            return self.test_data_sets[0].input_data.grad
        else:
            return self.test_data_inputs.\
                       grad[snapshot.grid_size*snapshot_number:
                            snapshot.grid_size*(snapshot_number+1)]

    def get_snapshot_calculation_output(self, snapshot_number):
        """
        Get the path to the output file for a specific snapshot.

        Parameters
        ----------
        snapshot_number : int
            Snapshot for which the calculation output should be returned.

        Returns
        -------
        calculation_output : string
            Path to the calculation output for this snapshot.

        """
        return self.parameters.snapshot_directories_list[snapshot_number].\
            calculation_output

    ##############################
    # Private methods
    ##############################

    # Loading data
    ######################

    def _check_snapshots(self): # ! TODO
        """Check the snapshots for consistency."""
        super(DataHandlerGraph, self)._check_snapshots()
        

        # Now we need to confirm that the snapshot list has some inner
        # consistency.
        if self.parameters.data_splitting_type == "by_snapshot":
            snapshot: Snapshot
            # As we are not actually interested in the number of snapshots,
            # but in the number of datasets, we also need to multiply by that.
            # for snapshot in self.parameters.snapshot_directories_list:
            #     if snapshot.snapshot_function == "tr":
            #         self.nr_training_snapshots += 1
            #     elif snapshot.snapshot_function == "te":
            #         self.nr_test_snapshots += 1
            #     elif snapshot.snapshot_function == "va":
            #         self.nr_validation_snapshots += 1
            #     else:
            #         raise Exception("Unknown option for snapshot splitting "
            #                         "selected.")
            self.nr_training_snapshots = len(self.raw_snapshots_train)
            self.nr_test_snapshots = len(self.raw_snapshots_test)
            self.nr_validation_snapshots = len(self.raw_snapshots_validation)

            # MALA can either be run in training or test-only mode.
            # But it has to be run in either of those!
            # So either training AND validation snapshots can be provided
            # OR only test snapshots.
            if self.nr_test_snapshots != 0:
                if self.nr_training_snapshots == 0:
                    printout("DataHandler prepared for inference. No training "
                             "possible with this setup. If this is not what "
                             "you wanted, please revise the input script. "
                             "Validation snapshots you may have entered will"
                             "be ignored.",
                             min_verbosity=0)
            else:
                if self.nr_training_snapshots == 0:
                    raise Exception("No training snapshots provided.")
                if self.nr_validation_snapshots == 0:
                    raise Exception("No validation snapshots provided.")
        else:
            raise Exception("Wrong parameter for data splitting provided.")



        # Reordering the lists.
        snapshot_order = {'tr': 0, 'va': 1, 'te': 2}
        self.parameters.snapshot_directories_list.sort(key=lambda d:
                                                       snapshot_order
                                                       [d.snapshot_function])


    def __build_datasets(self):
        """Build the DataSets that are used during training."""
        if self.parameters.use_lazy_loading:
            raise Exception("Lazy loading not supported in this mode")

        if self.parameters.use_clustering:
            raise Exception("Clustering not supported in this mode")
        
        # if self.input_data_scaler is not None:
        #     raise Exception("Data scalers not supported in this mode")
        
        # if self.output_data_scaler is not None:
        #     raise Exception("Data scalers not supported in this mode")

        if not self.parameters.use_graph_data_set:
            raise Exception("Wrong dataset type selected.")



        if self.nr_training_snapshots != 0:
            printout("Using GraphDataset.", min_verbosity=2)
            train_ldos_paths = [snapshot["ldos_path"] for snapshot in self.raw_snapshots_train]
            train_input_paths = [snapshot["input_path"] for snapshot in self.raw_snapshots_train]
            ldos_shape = self.raw_snapshots_train[0]["ldos_shape"]
            self.training_data_sets.append(GraphDataset(
                self.params.data.n_closest_ions, self.params.data.n_closest_ldos,
                self.params.running.ldos_grid_batch_size,
                ldos_paths=train_ldos_paths, input_paths=train_input_paths,
                ldos_shape=ldos_shape
            ))

        if self.nr_validation_snapshots != 0:
            printout("Using GraphDataset.", min_verbosity=2)
            validation_ldos_paths = [snapshot["ldos_path"] for snapshot in self.raw_snapshots_validation]
            validation_input_paths = [snapshot["input_path"] for snapshot in self.raw_snapshots_validation]
            ldos_shape = self.raw_snapshots_validation[0]["ldos_shape"]
            self.validation_data_sets.append(GraphDataset(
                self.params.data.n_closest_ions, self.params.data.n_closest_ldos,
                self.params.running.ldos_grid_batch_size,
                ldos_paths=validation_ldos_paths, input_paths=validation_input_paths,
                ldos_shape=ldos_shape
            ))

        if self.nr_test_snapshots != 0:
            printout("Using GraphDataset.", min_verbosity=2)
            test_ldos_paths = [snapshot["ldos_path"] for snapshot in self.raw_snapshots_test]
            test_input_paths = [snapshot["input_path"] for snapshot in self.raw_snapshots_test]
            ldos_shape = self.raw_snapshots_test[0]["ldos_shape"]
            self.test_data_inputs.requires_grad = True
            self.test_data_sets.append(GraphDataset(
                self.params.data.n_closest_ions, self.params.data.n_closest_ldos,
                self.params.running.ldos_grid_batch_size,
                ldos_paths=test_ldos_paths, input_paths=test_input_paths,
                ldos_shape=ldos_shape
            ))

    # Scaling
    ######################
    # Not implemented yet.
    
    
