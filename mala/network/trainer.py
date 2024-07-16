"""Trainer class for training a network."""

import os
import time
from datetime import datetime
from packaging import version

from mala.datahandling.on_the_fly_graph_dataset import Subset

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mala.common.parameters import printout
from mala.common.parallelizer import get_local_rank
from mala.datahandling.fast_tensor_dataset import FastTensorDataset
from mala.network.network import Network
from mala.network.runner import RunnerMLP, RunnerGraph
from mala.datahandling.lazy_load_dataset_single import LazyLoadDatasetSingle
from mala.datahandling.multi_lazy_load_data_loader import (
    MultiLazyLoadDataLoader,
)
from mala.datahandling.graph_dataset import GraphDataset
from dgl.dataloading import GraphDataLoader
from tqdm.auto import tqdm, trange

import dgl

from tqdm.auto import tqdm, trange


class Trainer:
    def __new__(cls, params, *args, **kwargs):
        if params.network.nn_type == "se3_transformer":
            return TrainerGNN(params, *args, **kwargs)
        else:
            return TrainerMLP(params, *args, **kwargs)


class TrainerMLP(RunnerMLP):
    """A class for training a neural network.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this TrainerMLP object.

    network : mala.network.network.Network
        Network which is being trained.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the training data.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, network, data, optimizer_dict=None):
        # copy the parameters into the class.
        super(TrainerMLP, self).__init__(params, network, data)

        if self.parameters_full.use_ddp:
            printout("DDP activated, wrapping model in DDP.", min_verbosity=1)
            # JOSHR: using streams here to maintain compatibility with
            # graph capture
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                self.network = DDP(self.network)
            torch.cuda.current_stream().wait_stream(s)

        self.final_test_loss = float("inf")
        self.initial_test_loss = float("inf")
        self.final_validation_loss = float("inf")
        self.initial_validation_loss = float("inf")
        self.optimizer = None
        self.scheduler = None
        self.patience_counter = 0
        self.last_epoch = 0
        self.last_loss = None
        self.training_data_loaders = []
        self.validation_data_loaders = []
        self.test_data_loaders = []

        # Samplers for the ddp case.
        self.train_sampler = None
        self.test_sampler = None
        self.validation_sampler = None

        self.__prepare_to_train(optimizer_dict)

        self.logger = None
        self.full_logging_path = None
        if self.parameters.logger is not None:
            os.makedirs(self.parameters.logging_dir, exist_ok=True)
            if self.parameters.logging_dir_append_date:
                date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                if len(self.parameters.run_name) > 0:
                    name = self.parameters.run_name + "_" + date_time
                else:
                    name = date_time
                self.full_logging_path = os.path.join(
                    self.parameters.logging_dir, name
                )
                os.makedirs(self.full_logging_path)
            else:
                self.full_logging_path = self.parameters.logging_dir

            # Set the path to log files
            if self.parameters.logger == "wandb":
                import wandb

                self.logger = wandb
            elif self.parameters.logger == "tensorboard":
                self.logger = SummaryWriter(self.full_logging_path)
            else:
                raise Exception(
                    f"Unsupported logger {self.parameters.logger}."
                )
            printout(
                "Writing logging output to",
                self.full_logging_path,
                min_verbosity=1,
            )

        self.gradscaler = None
        if self.parameters.use_mixed_precision:
            printout("Using mixed precision via AMP.", min_verbosity=1)
            self.gradscaler = torch.cuda.amp.GradScaler()

        self.train_graph = None
        self.validation_graph = None

    @classmethod
    def run_exists(cls, run_name, params_format="json", zip_run=True):
        """
        Check if a hyperparameter optimization checkpoint exists.

        Returns True if it does.

        Parameters
        ----------
        run_name : string
            Name of the checkpoint.

        params_format : bool
            Save format of the parameters.

        Returns
        -------
        checkpoint_exists : bool
            True if the checkpoint exists, False otherwise.

        """
        if zip_run is True:
            return os.path.isfile(run_name + ".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params." + params_format
            optimizer_name = run_name + ".optimizer.pth"
            return all(
                map(
                    os.path.isfile,
                    [
                        iscaler_name,
                        oscaler_name,
                        param_name,
                        network_name,
                        optimizer_name,
                    ],
                )
            )

    @classmethod
    def load_run(
        cls,
        run_name,
        path="./",
        zip_run=True,
        params_format="json",
        load_runner=True,
        prepare_data=True,
    ):
        """
        Load a run.

        Parameters
        ----------
        run_name : str
            Name under which the run is saved.

        path : str
            Path where the run is saved.

        zip_run : bool
            If True, MALA will attempt to load from a .zip file. If False,
            then separate files will be attempted to be loaded.

        params_format : str
            Can be "json" or "pkl", depending on what was saved by the model.
            Default is "json".

        load_runner : bool
            If True, a Runner object will be created/loaded for further use.

        prepare_data : bool
            If True, the data will be loaded into memory. This is needed when
            continuing a model training.

        Return
        ------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved to file.

        loaded_network : mala.network.network.Network
            The network saved to file.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from file.

        new_trainer : TrainerMLP
            (Optional) The runner reconstructed from file. For Tester and
            Predictor class, this is just a newly instantiated object.
        """
        return super(Trainer, cls).load_run(
            run_name,
            path=path,
            zip_run=zip_run,
            params_format=params_format,
            load_runner=load_runner,
            prepare_data=prepare_data,
            load_with_gpu=None,
            load_with_mpi=None,
            load_with_ddp=None,
        )

    @classmethod
    def _load_from_run(cls, params, network, data, file=None):
        """
        Load a trainer from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the trainer should be created.
            Has to be compatible with network and data.

        file : string
            Path to the file from which the trainer should be loaded.

        network : mala.network.network.Network
            Network which is being trained.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the training data.


        Returns
        -------
        loaded_trainer : Network
            The trainer that was loaded from the file.
        """
        # First, load the checkpoint.
        if params.use_ddp:
            map_location = {"cuda:%d" % 0: "cuda:%d" % get_local_rank()}
            checkpoint = torch.load(file, map_location=map_location)
        else:
            checkpoint = torch.load(file)

        # Now, create the Trainer class with it.
        loaded_trainer = Trainer(
            params, network, data, optimizer_dict=checkpoint
        )
        return loaded_trainer

    def train_network(self):
        """Train a network using data given by a DataHandler."""
        ############################
        # CALCULATE INITIAL METRICS
        ############################

        vloss = float("inf")
        self.initial_validation_loss = vloss

        # Initialize all the counters.
        checkpoint_counter = 0

        # If we restarted from a checkpoint, we have to differently initialize
        # the loss.
        if self.last_loss is None:
            vloss_old = vloss
        else:
            vloss_old = self.last_loss

        ############################
        # PERFORM TRAINING
        ############################

        total_batch_id = 0

        for epoch in range(self.last_epoch, self.parameters.max_number_epochs):
            start_time = time.time()

            # Prepare model for training.
            self.network.train()

            training_loss_sum_logging = 0.0

            # Process each mini batch and save the training loss.
            training_loss_sum = torch.zeros(
                1, device=self.parameters._configuration["device"]
            )

            # train sampler
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)

            # shuffle dataset if necessary
            if isinstance(self.data.training_data_sets[0], FastTensorDataset):
                self.data.training_data_sets[0].shuffle()

            if self.parameters._configuration["gpu"] > 0:
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )
                tsample = time.time()
                t0 = time.time()
                batchid = 0
                for loader in self.training_data_loaders:
                    t = time.time()
                    for inputs, outputs in tqdm(
                        loader,
                        desc="training",
                        disable=self.parameters_full.verbosity < 2,
                        total=len(loader),
                    ):
                        dt = time.time() - t
                        printout(f"load time: {dt}", min_verbosity=3)

                        if self.parameters.profiler_range is not None:
                            if batchid == self.parameters.profiler_range[0]:
                                torch.cuda.profiler.start()
                            if batchid == self.parameters.profiler_range[1]:
                                torch.cuda.profiler.stop()

                        torch.cuda.nvtx.range_push(f"step {batchid}")

                        torch.cuda.nvtx.range_push("data copy in")
                        t = time.time()
                        inputs = inputs.to(
                            self.parameters._configuration["device"],
                            non_blocking=True,
                        )
                        outputs = outputs.to(
                            self.parameters._configuration["device"],
                            non_blocking=True,
                        )
                        dt = time.time() - t
                        printout(f"data copy in time: {dt}", min_verbosity=3)
                        # data copy in
                        torch.cuda.nvtx.range_pop()

                        loss = self.__process_mini_batch(
                            self.network, inputs, outputs
                        )
                        # step
                        torch.cuda.nvtx.range_pop()
                        training_loss_sum += loss
                        training_loss_sum_logging += loss.item()

                        if (
                            batchid != 0
                            and (batchid + 1)
                            % self.parameters.training_log_interval
                            == 0
                        ):
                            torch.cuda.synchronize()
                            sample_time = time.time() - tsample
                            avg_sample_time = (
                                sample_time
                                / self.parameters.training_log_interval
                            )
                            avg_sample_tput = (
                                self.parameters.training_log_interval
                                * inputs.shape[0]
                                / sample_time
                            )
                            printout(
                                f"batch {batchid + 1}, "  # /{total_samples}, "
                                f"train avg time: {avg_sample_time} "
                                f"train avg throughput: {avg_sample_tput}",
                                min_verbosity=3,
                            )
                            tsample = time.time()

                            # summary_writer tensor board
                            if self.parameters.logger == "tensorboard":
                                training_loss_mean = (
                                    training_loss_sum_logging
                                    / self.parameters.training_log_interval
                                )
                                self.logger.add_scalars(
                                    "ldos",
                                    {"during_training": training_loss_mean},
                                    total_batch_id,
                                )
                                self.logger.close()
                                training_loss_sum_logging = 0.0
                            if self.parameters.logger == "wandb":
                                training_loss_mean = (
                                    training_loss_sum_logging
                                    / self.parameters.training_log_interval
                                )
                                self.logger.log(
                                    {
                                        "ldos_during_training": training_loss_mean
                                    },
                                    step=total_batch_id,
                                )
                                training_loss_sum_logging = 0.0

                        batchid += 1
                        total_batch_id += 1
                        t = time.time()
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )
                t1 = time.time()
                printout(f"training time: {t1 - t0}", min_verbosity=2)

                training_loss = training_loss_sum.item() / batchid

                # Calculate the validation loss. and output it.
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )
            else:
                batchid = 0
                for loader in self.training_data_loaders:
                    for inputs, outputs in loader:
                        inputs = inputs.to(
                            self.parameters._configuration["device"]
                        )
                        outputs = outputs.to(
                            self.parameters._configuration["device"]
                        )
                        training_loss_sum += self.__process_mini_batch(
                            self.network, inputs, outputs
                        )
                        batchid += 1
                training_loss = training_loss_sum.item() / batchid
            dataset_fractions = ["validation"]
            if self.parameters.validate_on_training_data:
                dataset_fractions.append("train")
            errors = self._validate_network(
                dataset_fractions, self.parameters.validation_metrics
            )
            for dataset_fraction in dataset_fractions:
                for metric in errors[dataset_fraction]:
                    errors[dataset_fraction][metric] = np.mean(
                        errors[dataset_fraction][metric]
                    )
            vloss = errors["validation"][
                self.parameters.during_training_metric
            ]
            if self.parameters_full.verbosity > 1:
                printout("Errors:", errors, min_verbosity=2)
            else:
                printout(
                    f"Epoch {epoch}: validation data loss: {vloss:.3e}",
                    min_verbosity=1,
                )

            if self.parameters.logger == "tensorboard":
                for dataset_fraction in dataset_fractions:
                    for metric in errors[dataset_fraction]:
                        self.logger.add_scalars(
                            metric,
                            {
                                dataset_fraction: errors[dataset_fraction][
                                    metric
                                ]
                            },
                            total_batch_id,
                        )
                self.logger.close()
            if self.parameters.logger == "wandb":
                for dataset_fraction in dataset_fractions:
                    for metric in errors[dataset_fraction]:
                        self.logger.log(
                            {
                                f"{dataset_fraction}_{metric}": errors[
                                    dataset_fraction
                                ][metric]
                            },
                            step=total_batch_id,
                        )

            if self.parameters._configuration["gpu"] > 0:
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )

            # Mix the DataSets up (this function only does something
            # in the lazy loading case).
            if self.parameters.use_shuffling_for_samplers:
                self.data.mix_datasets()
            if self.parameters._configuration["gpu"] > 0:
                torch.cuda.synchronize(
                    self.parameters._configuration["device"]
                )

            # If a scheduler is used, update it.
            if self.scheduler is not None:
                if (
                    self.parameters.learning_rate_scheduler
                    == "ReduceLROnPlateau"
                ):
                    self.scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (
                    1.0 - self.parameters.early_stopping_threshold
                ):
                    self.patience_counter = 0
                    vloss_old = vloss
                else:
                    self.patience_counter += 1
                    printout(
                        "Validation accuracy has not improved enough.",
                        min_verbosity=1,
                    )
                    if (
                        self.patience_counter
                        >= self.parameters.early_stopping_epochs
                    ):
                        printout(
                            "Stopping the training, validation "
                            "accuracy has not improved for",
                            self.patience_counter,
                            "epochs.",
                            min_verbosity=1,
                        )
                        self.last_epoch = epoch
                        break

            # If checkpointing is enabled, we need to checkpoint.
            if self.parameters.checkpoints_each_epoch != 0:
                checkpoint_counter += 1
                if (
                    checkpoint_counter
                    >= self.parameters.checkpoints_each_epoch
                ):
                    printout("Checkpointing training.", min_verbosity=0)
                    self.last_epoch = epoch
                    self.last_loss = vloss_old
                    self.__create_training_checkpoint()
                    checkpoint_counter = 0

            printout(
                "Time for epoch[s]:", time.time() - start_time, min_verbosity=2
            )

        # Clean-up for pre-fetching lazy loading.
        if self.data.parameters.use_lazy_loading_prefetch:
            self.training_data_loaders.cleanup()
            self.validation_data_loaders.cleanup()
            if len(self.data.test_data_sets) > 0:
                self.test_data_loaders.cleanup()

    def _validate_network(self, data_set_fractions, metrics):
        # """Validate a network, using train, test or validation data."""
        self.network.eval()
        errors = {}
        for data_set_type in data_set_fractions:
            if data_set_type == "train":
                data_loaders = self.training_data_loaders
                data_sets = self.data.training_data_sets
                number_of_snapshots = self.data.nr_training_snapshots
                offset_snapshots = 0

            elif data_set_type == "validation":
                data_loaders = self.validation_data_loaders
                data_sets = self.data.validation_data_sets
                number_of_snapshots = self.data.nr_validation_snapshots
                offset_snapshots = self.data.nr_training_snapshots

            elif data_set_type == "test":
                data_loaders = self.test_data_loaders
                data_sets = self.data.test_data_sets
                number_of_snapshots = self.data.nr_test_snapshots
                offset_snapshots = (
                    self.data.nr_validation_snapshots
                    + self.data.nr_training_snapshots
                )
            else:
                raise Exception(
                    f"Dataset type ({data_set_type}) not recognized."
                )

            errors[data_set_type] = {}
            for metric in metrics:
                errors[data_set_type][metric] = []

            if isinstance(data_loaders[0], MultiLazyLoadDataLoader):
                raise Exception("MultiLazyLoadDataLoader not supported.")

            with torch.no_grad():
                for snapshot_number in trange(
                    offset_snapshots,
                    number_of_snapshots + offset_snapshots,
                    desc="Validation",
                ):
                    # Get optimal batch size and number of batches per snapshotss
                    grid_size = self.data.parameters.snapshot_directories_list[
                        snapshot_number
                    ].grid_size

                    optimal_batch_size = self._correct_batch_size_for_testing(
                        grid_size, self.parameters.mini_batch_size
                    )
                    number_of_batches_per_snapshot = int(
                        grid_size / optimal_batch_size
                    )

                    actual_outputs, predicted_outputs = (
                        self._forward_entire_snapshot(
                            snapshot_number,
                            data_sets[0],
                            data_set_type[0:2],
                            number_of_batches_per_snapshot,
                            optimal_batch_size,
                        )
                    )

                    if "ldos" in metrics:
                        error = (
                            (actual_outputs - predicted_outputs) ** 2
                        ).mean()
                        errors[data_set_type]["ldos"].append(error)

                    energy_metrics = [
                        metric for metric in metrics if "energy" in metric
                    ]
                    if len(energy_metrics) > 0:
                        energy_errors = self._calculate_energy_errors(
                            actual_outputs,
                            predicted_outputs,
                            energy_metrics,
                            snapshot_number,
                        )
                        for metric in energy_metrics:
                            errors[data_set_type][metric].append(
                                energy_errors[metric]
                            )

                    if "number_of_electrons" in metrics:
                        if len(energy_metrics) == 0:
                            raise Exception(
                                "Number of electrons can only be calculated if energy metrics are calculated."
                            )
                        num_electrons = (
                            self.data.target_calculator.number_of_electrons_exact
                        )
                        num_electrons_pred = self.data.target_calculator.get_number_of_electrons(
                            predicted_outputs
                        )
                        error = abs(num_electrons - num_electrons_pred)
                        errors[data_set_type]["number_of_electrons"].append(
                            error
                        )
        return errors

    def __prepare_to_train(self, optimizer_dict):
        """Prepare everything for training."""
        # Configure keyword arguments for DataSampler.
        kwargs = {
            "num_workers": self.parameters.num_workers,
            "pin_memory": False,
        }
        if self.parameters_full.use_gpu > 0:
            kwargs["pin_memory"] = True

        # Read last epoch
        if optimizer_dict is not None:
            self.last_epoch = optimizer_dict["epoch"] + 1

        # Scale the learning rate according to ddp.
        if self.parameters_full.use_ddp:
            if dist.get_world_size() > 1 and self.last_epoch == 0:
                printout(
                    "Rescaling learning rate because multiple workers are"
                    " used for training.",
                    min_verbosity=1,
                )
                self.parameters.learning_rate = (
                    self.parameters.learning_rate * dist.get_world_size()
                )

        # Choose an optimizer to use.
        if self.parameters.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=self.parameters.learning_rate,
                weight_decay=self.parameters.l2_regularization,
            )
        elif self.parameters.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.parameters.learning_rate,
                weight_decay=self.parameters.l2_regularization,
            )
        elif self.parameters.optimizer == "FusedAdam":
            if version.parse(torch.__version__) >= version.parse("1.13.0"):
                self.optimizer = optim.Adam(
                    self.network.parameters(),
                    lr=self.parameters.learning_rate,
                    weight_decay=self.parameters.l2_regularization,
                    fused=True,
                )
            else:
                raise Exception("Optimizer requires " "at least torch 1.13.0.")
        else:
            raise Exception("Unsupported optimizer.")

        # Load data from pytorch file.
        if optimizer_dict is not None:
            self.optimizer.load_state_dict(
                optimizer_dict["optimizer_state_dict"]
            )
            self.patience_counter = optimizer_dict["early_stopping_counter"]
            self.last_loss = optimizer_dict["early_stopping_last_loss"]

        if self.parameters_full.use_ddp:
            # scaling the batch size for multiGPU per node
            # self.batch_size= self.batch_size*hvd.local_size()

            # If lazy loading is used we do not shuffle the data points on
            # their own, but rather shuffle them
            # by shuffling the files themselves and then reading file by file
            # per epoch.
            # This shuffling is done in the dataset themselves.
            do_shuffle = self.parameters.use_shuffling_for_samplers
            if self.data.parameters.use_lazy_loading:
                do_shuffle = False

            self.train_sampler = (
                torch.utils.data.distributed.DistributedSampler(
                    self.data.training_data_sets[0],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=do_shuffle,
                )
            )
            self.validation_sampler = (
                torch.utils.data.distributed.DistributedSampler(
                    self.data.validation_data_sets[0],
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                )
            )

            if self.data.test_data_sets:
                self.test_sampler = (
                    torch.utils.data.distributed.DistributedSampler(
                        self.data.test_data_sets[0],
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=False,
                    )
                )

        # Instantiate the learning rate scheduler, if necessary.
        if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=self.parameters.learning_rate_patience,
                mode="min",
                factor=self.parameters.learning_rate_decay,
                verbose=True,
            )
        elif self.parameters.learning_rate_scheduler is None:
            pass
        else:
            raise Exception("Unsupported learning rate schedule.")
        if self.scheduler is not None and optimizer_dict is not None:
            self.scheduler.load_state_dict(
                optimizer_dict["lr_scheduler_state_dict"]
            )

        # If lazy loading is used we do not shuffle the data points on their
        # own, but rather shuffle them
        # by shuffling the files themselves and then reading file by file per
        # epoch.
        # This shuffling is done in the dataset themselves.
        do_shuffle = self.parameters.use_shuffling_for_samplers
        if (
            self.data.parameters.use_lazy_loading
            or self.parameters_full.use_ddp
        ):
            do_shuffle = False

        # Prepare data loaders.(look into mini-batch size)
        if isinstance(self.data.training_data_sets[0], FastTensorDataset):
            # Not shuffling in loader.
            # I manually shuffle the data set each epoch.
            self.training_data_loaders.append(
                DataLoader(
                    self.data.training_data_sets[0],
                    batch_size=None,
                    sampler=self.train_sampler,
                    **kwargs,
                    shuffle=False,
                )
            )
        else:
            if isinstance(
                self.data.training_data_sets[0], LazyLoadDatasetSingle
            ):
                self.training_data_loaders = MultiLazyLoadDataLoader(
                    self.data.training_data_sets, **kwargs
                )
            else:
                self.training_data_loaders.append(
                    DataLoader(
                        self.data.training_data_sets[0],
                        batch_size=self.parameters.mini_batch_size,
                        sampler=self.train_sampler,
                        **kwargs,
                        shuffle=do_shuffle,
                    )
                )
        if isinstance(self.data.validation_data_sets[0], FastTensorDataset):
            self.validation_data_loaders.append(
                DataLoader(
                    self.data.validation_data_sets[0],
                    batch_size=None,
                    sampler=self.validation_sampler,
                    **kwargs,
                )
            )
        else:
            if isinstance(
                self.data.validation_data_sets[0], LazyLoadDatasetSingle
            ):
                self.validation_data_loaders = MultiLazyLoadDataLoader(
                    self.data.validation_data_sets, **kwargs
                )
            else:
                self.validation_data_loaders.append(
                    DataLoader(
                        self.data.validation_data_sets[0],
                        batch_size=self.parameters.mini_batch_size * 1,
                        sampler=self.validation_sampler,
                        **kwargs,
                    )
                )
        if self.data.test_data_sets:
            if isinstance(self.data.test_data_sets[0], LazyLoadDatasetSingle):
                self.test_data_loaders = MultiLazyLoadDataLoader(
                    self.data.test_data_sets, **kwargs
                )
            else:
                self.test_data_loaders.append(
                    DataLoader(
                        self.data.test_data_sets[0],
                        batch_size=self.parameters.mini_batch_size * 1,
                        sampler=self.test_sampler,
                        **kwargs,
                    )
                )

        if self.parameters_full.use_gpu > 1:
            if self.parameters_full.network.nn_type != "feed-forward":
                raise Exception(
                    "Only feed-forward networks are supported "
                    "with multiple GPUs."
                )
            self.network = torch.nn.DataParallel(
                self.network,
                device_ids=list(range(self.parameters_full.use_gpu)),
            )

    def __process_mini_batch(self, network, input_data, target_data):
        """Process a mini batch."""
        if self.parameters._configuration["gpu"] > 0:
            if self.parameters.use_graphs and self.train_graph is None:
                printout("Capturing CUDA graph for training.", min_verbosity=2)
                s = torch.cuda.Stream(self.parameters._configuration["device"])
                s.wait_stream(
                    torch.cuda.current_stream(
                        self.parameters._configuration["device"]
                    )
                )
                # Warmup for graphs
                with torch.cuda.stream(s):
                    for _ in range(20):
                        self.network.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast(
                            enabled=self.parameters.use_mixed_precision
                        ):
                            prediction = network(input_data)
                            if self.parameters_full.use_ddp:
                                # JOSHR: We have to use "module" here to access custom method of DDP wrapped model
                                loss = network.module.calculate_loss(
                                    prediction, target_data
                                )
                            else:
                                loss = network.calculate_loss(
                                    prediction, target_data
                                )

                        if self.gradscaler:
                            self.gradscaler.scale(loss).backward()
                        else:
                            loss.backward()
                torch.cuda.current_stream(
                    self.parameters._configuration["device"]
                ).wait_stream(s)

                # Create static entry point tensors to graph
                self.static_input_data = torch.empty_like(input_data)
                self.static_target_data = torch.empty_like(target_data)

                # Capture graph
                self.train_graph = torch.cuda.CUDAGraph()
                network.zero_grad(set_to_none=True)
                with torch.cuda.graph(self.train_graph):
                    with torch.cuda.amp.autocast(
                        enabled=self.parameters.use_mixed_precision
                    ):
                        self.static_prediction = network(
                            self.static_input_data
                        )

                        if self.parameters_full.use_ddp:
                            self.static_loss = network.module.calculate_loss(
                                self.static_prediction, self.static_target_data
                            )
                        else:
                            self.static_loss = network.calculate_loss(
                                self.static_prediction, self.static_target_data
                            )

                        if hasattr(network, "module"):
                            self.static_loss = network.module.calculate_loss(
                                self.static_prediction, self.static_target_data
                            )
                        else:
                            self.static_loss = network.calculate_loss(
                                self.static_prediction, self.static_target_data
                            )

                    if self.gradscaler:
                        self.gradscaler.scale(self.static_loss).backward()
                    else:
                        self.static_loss.backward()

            if self.train_graph:
                self.static_input_data.copy_(input_data)
                self.static_target_data.copy_(target_data)
                self.train_graph.replay()
            else:
                torch.cuda.nvtx.range_push("zero_grad")
                self.network.zero_grad(set_to_none=True)
                # zero_grad
                torch.cuda.nvtx.range_pop()

                with torch.cuda.amp.autocast(
                    enabled=self.parameters.use_mixed_precision
                ):
                    torch.cuda.nvtx.range_push("forward")
                    t = time.time()
                    prediction = network(input_data)
                    dt = time.time() - t
                    printout(f"forward time: {dt}", min_verbosity=3)
                    # forward
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("loss")
                    if self.parameters_full.use_ddp:
                        loss = network.module.calculate_loss(
                            prediction, target_data
                        )
                    else:
                        loss = network.calculate_loss(prediction, target_data)
                    dt = time.time() - t
                    printout(f"loss time: {dt}", min_verbosity=3)
                    # loss
                    torch.cuda.nvtx.range_pop()

                if self.gradscaler:
                    self.gradscaler.scale(loss).backward()
                else:
                    loss.backward()

            t = time.time()
            torch.cuda.nvtx.range_push("optimizer")
            if self.gradscaler:
                self.gradscaler.step(self.optimizer)
                self.gradscaler.update()
            else:
                self.optimizer.step()
            dt = time.time() - t
            printout(f"optimizer time: {dt}", min_verbosity=3)
            torch.cuda.nvtx.range_pop()  # optimizer

            if self.train_graph:
                return self.static_loss
            else:
                return loss
        else:
            prediction = network(input_data)
            if self.parameters_full.use_ddp:
                loss = network.module.calculate_loss(prediction, target_data)
            else:
                loss = network.calculate_loss(prediction, target_data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss

    def __create_training_checkpoint(self):
        """
        Create a checkpoint during training.

        Follows https://pytorch.org/tutorials/recipes/recipes/saving_and_
        loading_a_general_checkpoint.html to some degree.
        """
        optimizer_name = self.parameters.checkpoint_name + ".optimizer.pth"

        # Next, we save all the other objects.

        if self.parameters_full.use_ddp:
            if dist.get_rank() != 0:
                return
        if self.scheduler is None:
            save_dict = {
                "epoch": self.last_epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "early_stopping_counter": self.patience_counter,
                "early_stopping_last_loss": self.last_loss,
            }
        else:
            save_dict = {
                "epoch": self.last_epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.scheduler.state_dict(),
                "early_stopping_counter": self.patience_counter,
                "early_stopping_last_loss": self.last_loss,
            }
        torch.save(
            save_dict, optimizer_name, _use_new_zipfile_serialization=False
        )

        self.save_run(
            self.parameters.checkpoint_name,
            save_runner=True,
            save_path=self.parameters.run_name,
        )

    @staticmethod
    def __average_validation(val, name):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val)
        avg_loss = hvd.allreduce(tensor, name=name, op=hvd.Average)
        return avg_loss.item()


class TrainerGNN(RunnerGraph):
    """A class for training a graph neural network.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this TrainerGNN object.

    network : mala.network.network.Network
        Network which is being trained.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the training data.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, network, data, optimizer_dict=None):
        # copy the parameters into the class.
        super(TrainerGNN, self).__init__(params, network, data)
        self.final_test_loss = float("inf")
        self.initial_test_loss = float("inf")
        self.final_validation_loss = float("inf")
        self.initial_validation_loss = float("inf")
        self.encoder_optimizer = None
        self.decoder_optimizer = None
        self.encoder_scheduler = None
        self.decoder_scheduler = None
        self.patience_counter = 0
        self.last_epoch = 0
        self.last_loss = None
        self.training_data_loaders = []
        self.validation_data_loaders = []
        self.test_data_loaders = []

        self.train_sampler = None
        self.test_sampler = None
        self.validation_sampler = None

        self.__prepare_to_train(optimizer_dict)

        self.logger = None
        self.full_logging_path = None
        if self.parameters.logger is not None:
            if not os.path.exists(self.parameters.logging_dir):
                os.makedirs(self.parameters.logging_dir)
            if self.parameters.logging_dir_append_date:
                date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                if len(self.parameters.run_name) > 0:
                    name = self.parameters.run_name + "_" + date_time
                else:
                    name = date_time
                self.full_logging_path = os.path.join(
                    self.parameters.logging_dir, name
                )
                os.makedirs(self.full_logging_path)
            else:
                self.full_logging_path = self.parameters.logging_dir

            # Set the path to log files
            if self.parameters.logger == "tensorboard":
                self.logger = SummaryWriter(self.full_logging_path)
            elif self.parameters.logger == "wandb":
                import wandb

                self.logger = wandb
            printout(
                "Writing logging output to",
                self.full_logging_path,
                min_verbosity=1,
            )

        self.gradscaler = None
        if self.parameters.use_mixed_precision:
            printout("Using mixed precision via AMP.", min_verbosity=1)
            self.gradscaler = torch.cuda.amp.GradScaler()

        self.train_graph = None
        self.validation_graph = None

    @classmethod
    def run_exists(cls, run_name, params_format="json", zip_run=True):
        """
        Check if a hyperparameter optimization checkpoint exists.

        Returns True if it does.

        Parameters
        ----------
        run_name : string
            Name of the checkpoint.

        params_format : bool
            Save format of the parameters.

        Returns
        -------
        checkpoint_exists : bool
            True if the checkpoint exists, False otherwise.

        """
        if zip_run is True:
            return os.path.isfile(run_name + ".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params." + params_format
            encoder_optimizer_name = run_name + ".encoder_optimizer.pth"
            decoder_optimizer_name = run_name + ".decoder_optimizer.pth"
            return all(
                map(
                    os.path.isfile,
                    [
                        iscaler_name,
                        oscaler_name,
                        param_name,
                        network_name,
                        encoder_optimizer_name,
                        decoder_optimizer_name,
                    ],
                )
            )

    @classmethod
    def load_run(
        cls,
        run_name,
        path="./",
        zip_run=True,
        params_format="json",
        load_runner=True,
        prepare_data=True,
    ):
        """
        Load a run.

        Parameters
        ----------
        run_name : str
            Name under which the run is saved.

        path : str
            Path where the run is saved.

        zip_run : bool
            If True, MALA will attempt to load from a .zip file. If False,
            then separate files will be attempted to be loaded.

        params_format : str
            Can be "json" or "pkl", depending on what was saved by the model.
            Default is "json".

        load_runner : bool
            If True, a Runner object will be created/loaded for further use.

        prepare_data : bool
            If True, the data will be loaded into memory. This is needed when
            continuing a model training.

        Return
        ------
        loaded_params : mala.common.parameters.Parameters
            The Parameters saved to file.

        loaded_network : mala.network.network.Network
            The network saved to file.

        new_datahandler : mala.datahandling.data_handler.DataHandler
            The data handler reconstructed from file.

        new_TrainerGNN : TrainerGNN
            (Optional) The runner reconstructed from file. For Tester and
            Predictor class, this is just a newly instantiated object.
        """
        return super(TrainerGNN, cls).load_run(
            run_name,
            path=path,
            zip_run=zip_run,
            params_format=params_format,
            load_runner=load_runner,
            prepare_data=prepare_data,
        )

    @classmethod
    def _load_from_run(cls, params, network, data, file=None):
        """
        Load a TrainerGNN from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the TrainerGNN should be created.
            Has to be compatible with network and data.

        file : string
            Path to the file from which the TrainerGNN should be loaded.

        network : mala.network.network.Network
            Network which is being trained.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the training data.


        Returns
        -------
        loaded_TrainerGNN : Network
            The TrainerGNN that was loaded from the file.
        """
        # First, load the checkpoint.
        checkpoint = torch.load(file)

        # Now, create the TrainerGNN class with it.
        loaded_TrainerGNN = TrainerGNN(
            params, network, data, optimizer_dict=checkpoint
        )
        return loaded_TrainerGNN

    def calculate_initial_metrics(self):
        ############################
        # CALCULATE INITIAL METRICS
        ############################

        tloss = float("inf")
        vloss = self._validate_network(
            self.network,
            "validation",
            self.parameters.after_before_training_metric,
        )

        printout(
            "Initial Guess - validation data loss: ", vloss, min_verbosity=1
        )
        if self.data.test_data_sets:
            printout(
                "Initial Guess - test data loss: ", tloss, min_verbosity=1
            )

        # Save losses for later use.
        self.initial_validation_loss = vloss
        self.initial_test_loss = tloss

        if self.parameters.logger == "tensorboard":
            self.logger.add_scalars(
                "Loss",
                {
                    f"validation_{self.parameters.during_training_metric}": vloss
                },
                0,
            )
        if self.parameters.logger == "wandb":
            self.logger.log(
                {
                    f"validation_{self.parameters.during_training_metric}": vloss
                },
                step=0,
            )
        self.logger.close()
        return vloss

    def train_network(self):
        """Train a network using data given by a DataHandler."""
        print("Warning! Skipping initial metrics calculation")

        # If we restarted from a checkpoint, we have to differently initialize
        # the loss.
        # vloss = self.calculate_initial_metrics()
        vloss = np.inf
        self.initial_validation_loss = vloss
        self.initial_test_loss = vloss
        if self.last_loss is None:
            vloss_old = vloss
        else:
            vloss_old = self.last_loss

        best_vloss = np.inf

        ############################
        # PERFORM TRAINING
        ############################

        total_batch_id = 0
        checkpoint_counter = 0

        snapshots_per_epoch_counter = 0
        if self.parameters.snapshots_per_epoch > 0:
            if (
                self.training_data_loaders[0].n_snapshots
                % self.parameters.snapshots_per_epoch
                != 0
            ):
                raise Exception(
                    "snapshots_per_epoch must be a divisor of the number of snapshots."
                )

        for epoch in range(self.last_epoch, self.parameters.max_number_epochs):
            start_time = time.time()

            # Prepare model for training.
            self.network.train()

            # Process each mini batch and save the training loss.
            # training_loss_sum = torch.zeros(1, device=self.parameters._configuration["device"])
            training_loss_sum = 0.0
            training_loss_sum_logging = 0.0

            # shuffle dataset if necessary
            if isinstance(self.data.training_data_sets[0], FastTensorDataset):
                self.data.training_data_sets[0].shuffle()

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize()
                tsample = time.time()
                t0 = time.time()
                batchid = 0
                assert (
                    len(self.training_data_loaders) == 1
                ), "Only one training data loader supported for now."

                for loader in self.training_data_loaders:
                    embedding_hash = None
                    embedding_extended = None
                    embedding_step_counter = 0

                    if self.parameters.snapshots_per_epoch > 0:
                        dataset_subset_start_index = (
                            snapshots_per_epoch_counter * loader.n_ldos_batches
                        )
                        dataset_subset_end_index = (
                            snapshots_per_epoch_counter
                            + self.parameters.snapshots_per_epoch
                        ) * loader.n_ldos_batches
                        snapshots_per_epoch_counter += (
                            self.parameters.snapshots_per_epoch
                        )
                        snapshots_per_epoch_counter %= loader.n_snapshots
                        loader = Subset(
                            loader,
                            start_index=dataset_subset_start_index,
                            end_index=dataset_subset_end_index,
                        )

                    graph_ions: dgl.DGLGraph
                    graph_grid: dgl.DGLGraph
                    for graph_ions, graph_grid in tqdm(
                        loader,
                        desc="training",
                        disable=self.parameters_full.verbosity < 2,
                        # total=len(loader)
                    ):
                        if batchid == self.parameters.profiler_range[0]:
                            torch.cuda.profiler.start()
                        if batchid == self.parameters.profiler_range[1]:
                            torch.cuda.profiler.stop()

                        torch.cuda.nvtx.range_push(f"step {batchid}")

                        graph_ions_hash = hash(
                            graph_ions.edata["rel_pos"][:100].numpy().tobytes()
                        )
                        if (
                            graph_ions_hash != embedding_hash
                            or embedding_step_counter
                            >= self.parameters.embedding_reuse_steps
                        ):
                            self.encoder_optimizer.step()
                            self.encoder_optimizer.zero_grad()
                            torch.cuda.nvtx.range_push("embedding calcuation")
                            embedding_extended = self._compute_embedding(
                                graph_ions, graph_grid
                            )
                            torch.cuda.nvtx.range_pop()
                            embedding_hash = graph_ions_hash
                            embedding_step_counter = 0
                        embedding_step_counter += 1

                        torch.cuda.nvtx.range_push("data copy in")
                        graph_grid = graph_grid.to(
                            self.parameters._configuration["device"],
                            non_blocking=True,
                        )
                        graph_ions = graph_ions.to(
                            self.parameters._configuration["device"],
                            non_blocking=True,
                        )  # ? Is this necessary?
                        # data copy in
                        torch.cuda.nvtx.range_pop()

                        train_loss = self.__process_mini_batch(
                            self.network,
                            graph_ions,
                            graph_grid,
                            embedding_extended,
                        )
                        # step
                        torch.cuda.nvtx.range_pop()
                        loss_float = train_loss.detach().cpu().item()
                        training_loss_sum += loss_float
                        training_loss_sum_logging += loss_float

                        if (
                            batchid != 0
                            and (batchid + 1)
                            % self.parameters.training_log_interval
                            == 0
                        ):
                            torch.cuda.synchronize()
                            sample_time = time.time() - tsample
                            avg_sample_time = (
                                sample_time
                                / self.parameters.training_log_interval
                            )
                            avg_sample_tput = (
                                self.parameters.training_log_interval
                                * (
                                    graph_grid.num_nodes()
                                    - graph_ions.num_nodes()
                                )
                                / sample_time
                            )
                            printout(
                                f"batch {batchid + 1}, "  # /{total_samples}, "
                                f"train avg time: {avg_sample_time} "
                                f"train avg throughput: {avg_sample_tput}",
                                min_verbosity=3,
                            )
                            tsample = time.time()

                            # summary_writer tensor board
                            training_loss_mean = (
                                training_loss_sum_logging
                                / self.parameters.training_log_interval
                            )
                            if self.parameters.logger == "tensorboard":
                                self.logger.add_scalars(
                                    "Loss",
                                    {"training": training_loss_mean},
                                    total_batch_id,
                                )
                                self.logger.close()
                            if self.parameters.logger == "wandb":
                                self.logger.log(
                                    {"training": training_loss_mean},
                                    step=total_batch_id,
                                )
                            training_loss_sum_logging = 0.0
                        batchid += 1
                        total_batch_id += 1
                torch.cuda.synchronize()
                t1 = time.time()
                printout(f"training time: {t1 - t0}", min_verbosity=2)

                # Calculate the validation loss. and output it.
                torch.cuda.synchronize()
            else:
                batchid = 0

                for loader in self.training_data_loaders:
                    graph_ions: dgl.DGLGraph
                    graph_grid: dgl.DGLGraph
                    embeddings = {}

                    for graph_ions, graph_grid in loader:
                        graph_ions_hash = hash(
                            graph_ions.edata["rel_pos"][:100].numpy().tobytes()
                        )
                        if graph_ions_hash not in embeddings:
                            torch.cuda.nvtx.range_push("embedding calcuation")
                            embedding_extended = self._compute_embedding(
                                graph_ions, graph_grid
                            )
                            torch.cuda.nvtx.range_pop()
                            embeddings[graph_ions_hash] = embedding_extended
                        embedding_extended = embeddings[graph_ions_hash]

                        train_loss = (
                            self.__process_mini_batch(
                                self.network,
                                graph_ions,
                                graph_grid,
                                embedding_extended,
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        training_loss_sum += train_loss
                        batchid += 1
                        total_batch_id += 1
                    self.encoder_optimizer.step()
                    self.encoder_optimizer.zero_grad()
                training_loss = training_loss_sum / batchid

            dataset_fractions = ["validation"]
            if self.parameters.validate_on_training_data:
                dataset_fractions.append("train")
            errors = self._validate_network(
                dataset_fractions, self.parameters.validation_metrics
            )
            for dataset_fraction in dataset_fractions:
                for metric in errors[dataset_fraction]:
                    errors[dataset_fraction][metric] = np.mean(
                        errors[dataset_fraction][metric]
                    )
            vloss = errors["validation"][
                self.parameters.during_training_metric
            ]
            if self.parameters_full.verbosity > 1:
                printout("Errors:", errors, min_verbosity=2)
            else:
                printout(
                    f"Epoch {epoch}: validation data loss: {vloss:.3e}",
                    min_verbosity=1,
                )

            if self.parameters.logger == "tensorboard":
                for dataset_fraction in dataset_fractions:
                    for metric in errors[dataset_fraction]:
                        self.logger.add_scalars(
                            metric,
                            {
                                dataset_fraction: errors[dataset_fraction][
                                    metric
                                ]
                            },
                            total_batch_id,
                        )
                self.logger.close()
            if self.parameters.logger == "wandb":
                for dataset_fraction in dataset_fractions:
                    for metric in errors[dataset_fraction]:
                        self.logger.log(
                            {
                                f"{dataset_fraction}_{metric}": errors[
                                    dataset_fraction
                                ][metric]
                            },
                            step=total_batch_id,
                        )

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize()

            # Mix the DataSets up (this function only does something
            # in the lazy loading case).
            if self.parameters.use_shuffling_for_samplers:
                self.data.mix_datasets()
            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize()

            # If a scheduler is used, update it.
            if self.encoder_scheduler is not None:
                if (
                    self.parameters.learning_rate_scheduler
                    == "ReduceLROnPlateau"
                ):
                    self.encoder_scheduler.step(vloss)
            if self.decoder_scheduler is not None:
                if (
                    self.parameters.learning_rate_scheduler
                    == "ReduceLROnPlateau"
                ):
                    self.decoder_scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (
                    1.0 - self.parameters.early_stopping_threshold
                ):
                    self.patience_counter = 0
                    vloss_old = vloss
                else:
                    self.patience_counter += 1
                    printout(
                        "Validation accuracy has not improved " "enough.",
                        min_verbosity=2,
                    )
                    if (
                        self.patience_counter
                        >= self.parameters.early_stopping_epochs
                    ):
                        printout(
                            "Stopping the training, validation "
                            "accuracy has not improved for",
                            self.patience_counter,
                            "epochs.",
                            min_verbosity=2,
                        )
                        self.last_epoch = epoch
                        break

            # If checkpointing is enabled, we need to checkpoint.
            if self.parameters.checkpoints_each_epoch != 0:
                checkpoint_counter += 1
                if (
                    checkpoint_counter
                    >= self.parameters.checkpoints_each_epoch
                ):
                    printout("Checkpointing training.", min_verbosity=0)
                    self.last_epoch = epoch
                    self.last_loss = vloss_old
                    self.__create_training_checkpoint(epoch=epoch)
                    checkpoint_counter = 0

            if self.parameters.checkpoint_best_so_far and vloss < best_vloss:
                printout(
                    f"Checkpointing training because of improved vloss {vloss}<{best_vloss}.",
                    min_verbosity=0,
                )
                self.last_epoch = epoch
                self.last_loss = vloss_old
                self.__create_training_checkpoint(epoch=epoch)
                best_vloss = vloss

            printout(
                "Time for epoch[s]:", time.time() - start_time, min_verbosity=2
            )

        ############################
        # CALCULATE FINAL METRICS
        ############################

        if (
            self.parameters.after_before_training_metric
            != self.parameters.during_training_metric
        ):
            vloss = self._validate_network(
                self.network,
                "validation",
                self.parameters.after_before_training_metric,
            )

        # Calculate final loss.
        self.final_validation_loss = vloss
        printout("Final validation data loss: ", vloss, min_verbosity=0)

        tloss = float("inf")
        if len(self.data.test_data_sets) > 0:
            tloss = self._validate_network(
                self.network,
                "test",
                self.parameters.after_before_training_metric,
            )
            printout("Final test data loss: ", tloss, min_verbosity=0)
        self.final_test_loss = tloss

        # Clean-up for pre-fetching lazy loading.
        if self.data.parameters.use_lazy_loading_prefetch:
            self.training_data_loaders.cleanup()
            self.validation_data_loaders.cleanup()
            if len(self.data.test_data_sets) > 0:
                self.test_data_loaders.cleanup()

    def _validate_network(self, data_set_fractions, metrics):
        # """Validate a network, using train, test or validation data."""
        self.network.eval()
        errors = {}
        for data_set_type in data_set_fractions:
            if data_set_type == "train":
                data_loaders = self.training_data_loaders
                data_sets = self.data.training_data_sets
                number_of_snapshots = self.data.nr_training_snapshots

            elif data_set_type == "validation":
                data_loaders = self.validation_data_loaders
                data_sets = self.data.validation_data_sets
                number_of_snapshots = self.data.nr_validation_snapshots

            elif data_set_type == "test":
                data_loaders = self.test_data_loaders
                data_sets = self.data.test_data_sets
                number_of_snapshots = self.data.nr_test_snapshots
            else:
                raise Exception(
                    f"Dataset type ({data_set_type}) not recognized."
                )

            errors[data_set_type] = {}
            for metric in metrics:
                errors[data_set_type][metric] = []

            if isinstance(data_loaders[0], MultiLazyLoadDataLoader):
                raise Exception("MultiLazyLoadDataLoader not supported.")

            with torch.no_grad():
                for snapshot_number in trange(
                    number_of_snapshots, desc="Validation"
                ):
                    # Get optimal batch size and number of batches per snapshotss
                    grid_size = self.data.parameters.snapshot_directories_list[
                        snapshot_number
                    ].grid_size

                    optimal_batch_size = self._correct_batch_size_for_testing(
                        grid_size, self.parameters.mini_batch_size
                    )
                    number_of_batches_per_snapshot = int(
                        grid_size / optimal_batch_size
                    )

                    actual_outputs, predicted_outputs = (
                        self._forward_entire_snapshot(
                            snapshot_number,
                            data_sets[0],
                            data_set_type[0:2],
                            number_of_batches_per_snapshot,
                            optimal_batch_size,
                        )
                    )

                    if "ldos" in metrics:
                        error = (
                            (actual_outputs - predicted_outputs) ** 2
                        ).mean()
                        errors[data_set_type]["ldos"].append(error)

                    energy_metrics = [
                        metric for metric in metrics if "energy" in metric
                    ]
                    if len(energy_metrics) > 0:
                        energy_errors = self._calculate_energy_errors(
                            actual_outputs,
                            predicted_outputs,
                            energy_metrics,
                            snapshot_number,
                        )
                        for metric in energy_errors.keys():
                            errors[data_set_type][metric].append(
                                energy_errors[metric]
                            )

                    if "number_of_electrons" in metrics:
                        if len(energy_metrics) == 0:
                            raise Exception(
                                "Number of electrons can only be calculated if some energy metrics are calculated."
                            )
                        num_electrons = (
                            self.data.target_calculator.number_of_electrons_exact
                        )
                        num_electrons_pred = self.data.target_calculator.get_number_of_electrons(
                            predicted_outputs
                        )
                        error = abs(num_electrons - num_electrons_pred)
                        errors[data_set_type]["number_of_electrons"].append(
                            error
                        )
        return errors

    def __prepare_to_train(self, optimizer_dict):
        """Prepare everything for training."""
        # Configure keyword arguments for DataSampler.
        kwargs = {
            "num_workers": self.parameters.num_workers,
            "pin_memory": False,
        }
        if self.parameters_full.use_gpu:
            kwargs["pin_memory"] = True

        # Read last epoch
        if optimizer_dict is not None:
            self.last_epoch = optimizer_dict["epoch"] + 1

        # Choose an optimizer to use.
        if self.parameters.optimizer == "SGD":
            raise Exception("SGD is not supported.")
        elif self.parameters.optimizer == "Adam":
            self.encoder_optimizer = optim.Adam(
                self.network.encoder.parameters(),
                lr=self.parameters.learning_rate_embedding,
                weight_decay=self.parameters.l2_regularization,
            )
            self.decoder_optimizer = optim.Adam(
                self.network.decoder.parameters(),
                lr=self.parameters.learning_rate,
                weight_decay=self.parameters.l2_regularization,
            )
        elif self.parameters.optimizer == "FusedAdam":
            raise Exception("FusedAdam is not supported.")
        else:
            raise Exception("Unsupported training method.")

        # Load data from pytorch file.
        if optimizer_dict is not None:
            raise Exception("Loading optimizer state is not supported.")
            self.optimizer.load_state_dict(
                optimizer_dict["optimizer_state_dict"]
            )
            self.patience_counter = optimizer_dict["early_stopping_counter"]
            self.last_loss = optimizer_dict["early_stopping_last_loss"]

        # Instantiate the learning rate scheduler, if necessary.
        if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
            # raise Exception("ReduceLROnPlateau is not supported.")
            self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.encoder_optimizer,
                patience=self.parameters.learning_rate_patience,
                mode="min",
                factor=self.parameters.learning_rate_decay,
                verbose=True,
            )
            self.decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.decoder_optimizer,
                patience=self.parameters.learning_rate_patience,
                mode="min",
                factor=self.parameters.learning_rate_decay,
                verbose=True,
            )
        elif self.parameters.learning_rate_scheduler is None:
            pass
        else:
            raise Exception("Unsupported learning rate scheduler.")
        if self.encoder_scheduler is not None and optimizer_dict is not None:
            self.encoder_scheduler.load_state_dict(
                optimizer_dict["lr_encoder_scheduler_state_dict"]
            )
        if self.decoder_scheduler is not None and optimizer_dict is not None:
            self.decoder_scheduler.load_state_dict(
                optimizer_dict["lr_decoder_scheduler_state_dict"]
            )

        # If lazy loading is used we do not shuffle the data points on their
        # own, but rather shuffle them
        # by shuffling the files themselves and then reading file by file per
        # epoch.
        # This shuffling is done in the dataset themselves.
        do_shuffle = self.parameters.use_shuffling_for_samplers
        if (
            self.data.parameters.use_lazy_loading
        ):
            do_shuffle = False

        # Prepare data loaders.(look into mini-batch size)
        if self.parameters_full.data.use_graph_data_set:
            self.training_data_loaders.append(self.data.training_data_sets[0])
        elif isinstance(self.data.training_data_sets[0], FastTensorDataset):
            # Not shuffling in loader.
            # I manually shuffle the data set each epoch.
            self.training_data_loaders.append(
                DataLoader(
                    self.data.training_data_sets[0],
                    batch_size=None,
                    sampler=self.train_sampler,
                    **kwargs,
                    shuffle=False,
                )
            )
        else:
            if isinstance(
                self.data.training_data_sets[0], LazyLoadDatasetSingle
            ):
                self.training_data_loaders = MultiLazyLoadDataLoader(
                    self.data.training_data_sets, **kwargs
                )
            else:
                self.training_data_loaders.append(
                    DataLoader(
                        self.data.training_data_sets[0],
                        batch_size=self.parameters.mini_batch_size,
                        sampler=self.train_sampler,
                        **kwargs,
                        shuffle=do_shuffle,
                    )
                )
        if self.parameters_full.data.use_graph_data_set:
            self.validation_data_loaders.append(
                GraphDataLoader(
                    self.data.validation_data_sets[0],
                    batch_size=None,
                    sampler=self.validation_sampler,
                    **kwargs,
                )
            )
        elif isinstance(self.data.validation_data_sets[0], FastTensorDataset):
            self.validation_data_loaders.append(
                DataLoader(
                    self.data.validation_data_sets[0],
                    batch_size=None,
                    sampler=self.validation_sampler,
                    **kwargs,
                )
            )
        else:
            if isinstance(
                self.data.validation_data_sets[0], LazyLoadDatasetSingle
            ):
                self.validation_data_loaders = MultiLazyLoadDataLoader(
                    self.data.validation_data_sets, **kwargs
                )
            else:
                self.validation_data_loaders.append(
                    DataLoader(
                        self.data.validation_data_sets[0],
                        batch_size=self.parameters.mini_batch_size * 1,
                        sampler=self.validation_sampler,
                        **kwargs,
                    )
                )
        if self.data.test_data_sets:
            if self.parameters_full.data.use_graph_data_set:
                self.test_data_loaders.append(
                    GraphDataLoader(
                        self.data.test_data_sets[0],
                        batch_size=None,
                        sampler=self.test_sampler,
                        **kwargs,
                    )
                )
            elif isinstance(
                self.data.test_data_sets[0], LazyLoadDatasetSingle
            ):
                self.test_data_loaders = MultiLazyLoadDataLoader(
                    self.data.test_data_sets, **kwargs
                )
            else:
                self.test_data_loaders.append(
                    DataLoader(
                        self.data.test_data_sets[0],
                        batch_size=self.parameters.mini_batch_size * 1,
                        sampler=self.test_sampler,
                        **kwargs,
                    )
                )

    def __process_mini_batch(
        self,
        network,
        graph_ions: dgl.DGLGraph,
        graph_grid: dgl.DGLGraph,
        embedding_extended: dict = None,
    ):
        """Process a mini batch."""
        if self.parameters._configuration["gpu"]:
            if self.parameters.use_graphs and self.train_graph is None:
                # ! Test this out
                # raise Exception("Not tested for now")
                printout("Capturing CUDA graph for training.", min_verbosity=2)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                # Warmup for graphs
                with torch.cuda.stream(s):
                    for i in range(20):
                        self.network.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast(
                            enabled=self.parameters.use_mixed_precision
                        ):
                            # prediction = network(graph_ions, graph_grid)
                            embedding_extended_ = network.get_embedding(
                                graph_ions, graph_grid
                            )
                            prediction = network.predict_ldos(
                                embedding_extended_, graph_ions, graph_grid
                            )
                            loss = network.calculate_loss(
                                prediction, graph_ions, graph_grid
                            )

                        if self.gradscaler:
                            self.gradscaler.scale(loss).backward()
                        else:
                            loss.backward()
                torch.cuda.current_stream().wait_stream(s)

                # Create static entry point tensors to graph
                self.static_graph_ions = graph_ions.clone()
                for key, value in graph_ions.ndata.items():
                    self.static_graph_ions.ndata[key] = value.clone()
                for key, value in graph_ions.edata.items():
                    self.static_graph_ions.edata[key] = value.clone()

                self.static_graph_grid = graph_grid.clone()
                for key, value in graph_grid.ndata.items():
                    self.static_graph_grid.ndata[key] = value.clone()
                for key, value in graph_grid.edata.items():
                    self.static_graph_grid.edata[key] = value.clone()

                self.static_embedding_extended = {}
                for key, value in embedding_extended.items():
                    self.static_embedding_extended[key] = value.clone()

                # Capture graph
                self.train_graph = torch.cuda.CUDAGraph()
                self.network.decoder.zero_grad(set_to_none=True)
                with torch.cuda.graph(self.train_graph):
                    with torch.cuda.amp.autocast(
                        enabled=self.parameters.use_mixed_precision
                    ):
                        self.static_prediction = network.predict_ldos(
                            self.static_embedding_extended,
                            self.static_graph_ions,
                            self.static_graph_grid,
                        )
                        self.static_loss = network.calculate_loss(
                            self.static_prediction,
                            self.static_graph_ions,
                            self.static_graph_grid,
                        )

                    if self.gradscaler:
                        self.gradscaler.scale(self.static_loss).backward()
                    else:
                        self.static_loss.backward()

            if self.train_graph:
                # ! Assumes same number of nodes and edges

                # Copy connections (edges) to static graph

                # Copy data to static tensors
                src_ions, dst_ions = graph_ions.edges()
                src_ions_static, dst_ions_static = (
                    self.static_graph_ions.edges()
                )
                src_ions_static.copy_(src_ions)
                dst_ions_static.copy_(dst_ions)
                for key, value in graph_ions.ndata.items():
                    self.static_graph_ions.ndata[key].copy_(value)
                for key, value in graph_ions.edata.items():
                    self.static_graph_ions.edata[key].copy_(value)

                src_grid, dst_grid = graph_grid.edges()
                src_grid_static, dst_grid_static = (
                    self.static_graph_grid.edges()
                )
                src_grid_static.copy_(src_grid)
                dst_grid_static.copy_(dst_grid)
                for key, value in graph_grid.ndata.items():
                    self.static_graph_grid.ndata[key].copy_(value)
                for key, value in graph_grid.edata.items():
                    self.static_graph_grid.edata[key].copy_(value)

                for key, value in embedding_extended.items():
                    self.static_embedding_extended[key].copy_(value)

                self.train_graph.replay()
            else:
                torch.cuda.nvtx.range_push("zero_grad")
                self.network.decoder.zero_grad(set_to_none=True)
                torch.cuda.nvtx.range_pop()

                with torch.cuda.amp.autocast(
                    enabled=self.parameters.use_mixed_precision
                ):
                    torch.cuda.nvtx.range_push("forward")
                    prediction = network.predict_ldos(
                        embedding_extended, graph_ions, graph_grid
                    )
                    # forward
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("loss")
                    loss = network.calculate_loss(
                        prediction, graph_ions, graph_grid
                    )
                    # loss
                    torch.cuda.nvtx.range_pop()

                if self.gradscaler:
                    self.gradscaler.scale(loss).backward()
                else:
                    self.decoder_optimizer.zero_grad(set_to_none=True)
                    torch.cuda.nvtx.range_push("backward")
                    loss.backward(retain_graph=True)
                    torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("optimizer")
            if self.gradscaler:
                self.gradscaler.step(self.decoder_optimizer)
                self.gradscaler.update()
            else:
                self.decoder_optimizer.step()
            torch.cuda.nvtx.range_pop()  # optimizer

            if self.train_graph:
                return self.static_loss
            else:
                return loss
        else:
            raise ValueError("CPU training not supported")
            # prediction = network(graph_ions, graph_grid)
            prediction = network.predict_ldos(
                embedding_extended, graph_ions, graph_grid
            )
            loss = network.calculate_loss(prediction, graph_ions, graph_grid)
            loss.backward()
            self.decoder_optimizer.step()
            self.decoder_optimizer.zero_grad()
            return loss

    def __create_training_checkpoint(self, epoch=None):
        """
        Create a checkpoint during training.

        Follows https://pytorch.org/tutorials/recipes/recipes/saving_and_
        loading_a_general_checkpoint.html to some degree.
        """
        encoder_optimizer_name = (
            self.parameters.checkpoint_name + ".encoder_optimizer.pth"
        )
        decoder_optimizer_name = (
            self.parameters.checkpoint_name + ".decoder_optimizer.pth"
        )

        # Next, we save all the other objects.
        if self.encoder_scheduler is None and self.decoder_scheduler is None:
            save_dict = {
                "epoch": self.last_epoch,
                "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
                "decoder_optimizer_state_dict": self.decoder_optimizer.state_dict(),
                "early_stopping_counter": self.patience_counter,
                "early_stopping_last_loss": self.last_loss,
            }
        else:
            save_dict = {
                "epoch": self.last_epoch,
                "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
                "decoder_optimizer_state_dict": self.decoder_optimizer.state_dict(),
                "lr_encoder_scheduler_state_dict": self.encoder_scheduler.state_dict(),
                "lr_decoder_scheduler_state_dict": self.decoder_scheduler.state_dict(),
                "early_stopping_counter": self.patience_counter,
                "early_stopping_last_loss": self.last_loss,
            }
        torch.save(
            save_dict,
            encoder_optimizer_name,
            _use_new_zipfile_serialization=True,
        )
        torch.save(
            save_dict,
            decoder_optimizer_name,
            _use_new_zipfile_serialization=True,
        )
        if epoch is not None:
            run_name = f"{self.parameters.checkpoint_name}_epoch_{epoch}"
        else:
            run_name = f"{self.parameters.checkpoint_name}"

        # self.save_run(run_name, save_runner=True)
        self.save_run(
            run_name, save_runner=False
        )  # ! TEMPORARY because breaking

    @staticmethod
    def __average_validation(val, name, device="cpu"):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val, device=device)
        dist.all_reduce(tensor)
        avg_loss = tensor / dist.get_world_size()
        return avg_loss.item()
