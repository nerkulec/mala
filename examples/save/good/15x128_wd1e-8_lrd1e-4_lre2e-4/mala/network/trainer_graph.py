"""TrainerGraph class for training a network."""
import os
import time
from datetime import datetime
from packaging import version

try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mala.common.parameters import printout
from mala.common.parallelizer import parallel_warn
from mala.datahandling.fast_tensor_dataset import FastTensorDataset
from mala.network.network import Network
from mala.network.runner_graph import RunnerGraph
from mala.datahandling.lazy_load_dataset_single import LazyLoadDatasetSingle
from mala.datahandling.multi_lazy_load_data_loader import \
    MultiLazyLoadDataLoader
from mala.datahandling.graph_dataset import GraphDataset
import dgl
from dgl.dataloading import GraphDataLoader
from tqdm.auto import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP


class TrainerGraph(RunnerGraph):
    """A class for training a graph neural network.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this TrainerGraph object.

    network : mala.network.network.Network
        Network which is being trained.

    data : mala.datahandling.data_handler.DataHandler
        DataHandler holding the training data.

    use_pkl_checkpoints : bool
        If true, .pkl checkpoints will be created.
    """

    def __init__(self, params, network, data, optimizer_dict=None):
        # copy the parameters into the class.
        super(TrainerGraph, self).__init__(params, network, data)
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

        # Samplers for the horovod case.
        self.train_sampler = None
        self.test_sampler = None
        self.validation_sampler = None

        self.__prepare_to_train(optimizer_dict)

        self.tensor_board = None
        self.full_visualization_path = None
        if self.parameters.visualisation:
            if not os.path.exists(self.parameters.visualisation_dir):
                os.makedirs(self.parameters.visualisation_dir)
            if self.parameters.visualisation_dir_append_date:
                date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                if len(self.parameters.run_name) > 0:
                    name = self.parameters.run_name + "_" + date_time
                else:
                    name = date_time
                self.full_visualization_path = \
                    os.path.join(self.parameters.visualisation_dir, name)
                os.makedirs(self.full_visualization_path)
            else:
                self.full_visualization_path = \
                    self.parameters.visualisation_dir

            # Set the path to log files
            self.tensor_board = SummaryWriter(self.full_visualization_path)
            printout("Writing visualization output to",
                     self.full_visualization_path, min_verbosity=1)

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
            return os.path.isfile(run_name+".zip")
        else:
            network_name = run_name + ".network.pth"
            iscaler_name = run_name + ".iscaler.pkl"
            oscaler_name = run_name + ".oscaler.pkl"
            param_name = run_name + ".params."+params_format
            encoder_optimizer_name = run_name + ".encoder_optimizer.pth"
            decoder_optimizer_name = run_name + ".decoder_optimizer.pth"
            return all(map(os.path.isfile, [
                iscaler_name, oscaler_name, param_name,
                network_name, encoder_optimizer_name, decoder_optimizer_name
            ]))

    @classmethod
    def load_run(cls, run_name, path="./", zip_run=True,
                 params_format="json", load_runner=True,
                 prepare_data=True):
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

        new_TrainerGraph : TrainerGraph
            (Optional) The runner reconstructed from file. For Tester and
            Predictor class, this is just a newly instantiated object.
        """
        return super(TrainerGraph, cls).load_run(run_name, path=path,
                                            zip_run=zip_run,
                                            params_format=params_format,
                                            load_runner=load_runner,
                                            prepare_data=prepare_data)

    @classmethod
    def _load_from_run(cls, params, network, data, file=None):
        """
        Load a TrainerGraph from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the TrainerGraph should be created.
            Has to be compatible with network and data.

        file : string
            Path to the file from which the TrainerGraph should be loaded.

        network : mala.network.network.Network
            Network which is being trained.

        data : mala.datahandling.data_handler.DataHandler
            DataHandler holding the training data.


        Returns
        -------
        loaded_TrainerGraph : Network
            The TrainerGraph that was loaded from the file.
        """
        # First, load the checkpoint.
        checkpoint = torch.load(file)

        # Now, create the TrainerGraph class with it.
        loaded_TrainerGraph = TrainerGraph(params, network, data,
                                 optimizer_dict=checkpoint)
        return loaded_TrainerGraph

    def calculate_initial_metrics(self):
        ############################
        # CALCULATE INITIAL METRICS
        ############################

        tloss = float("inf")
        vloss = self.__validate_network(
            self.network, "validation",
            self.parameters.after_before_training_metric
        )

        if self.data.test_data_sets:
            tloss = self.__validate_network(
                self.network, "test",
                self.parameters.after_before_training_metric
            )

        # Collect and average all the losses from all the devices
        if self.parameters_full.use_horovod:
            vloss = self.__average_validation(vloss, 'average_loss')
            self.initial_validation_loss = vloss
            if self.data.test_data_set is not None:
                tloss = self.__average_validation(tloss, 'average_loss')
                self.initial_test_loss = tloss

        printout("Initial Guess - validation data loss: ", vloss,
                 min_verbosity=1)
        if self.data.test_data_sets:
            printout("Initial Guess - test data loss: ", tloss,
                     min_verbosity=1)

        # Save losses for later use.
        self.initial_validation_loss = vloss
        self.initial_test_loss = tloss


        if self.parameters.visualisation:
            self.tensor_board.add_scalars(
                'Loss', {'validation': vloss}, 0
            )
            self.tensor_board.close()

        # If we restarted from a checkpoint, we have to differently initialize
        # the loss.
        if self.last_loss is None:
            vloss_old = vloss
        else:
            vloss_old = self.last_loss


    def train_network(self):
        """Train a network using data given by a DataHandler."""
        print("Warning! Skipping initial metrics calculation")
        # self.calculate_initial_metrics()

        ############################
        # PERFORM TRAINING
        ############################

        # self.network.encoder = DDP(self.network.encoder)
        # self.network.decoder = DDP(self.network.decoder)

        total_batch_id = 0

        for epoch in range(self.last_epoch, self.parameters.max_number_epochs):
            start_time = time.time()

            # Prepare model for training.
            self.network.train()

            # Process each mini batch and save the training loss.
            # training_loss_sum = torch.zeros(1, device=self.parameters._configuration["device"])
            training_loss_sum = 0.0
            training_loss_sum_logging = 0.0

            # train sampler
            if self.parameters_full.use_horovod:
                self.train_sampler.set_epoch(epoch)

            # shuffle dataset if necessary
            if isinstance(self.data.training_data_sets[0], FastTensorDataset):
                self.data.training_data_sets[0].shuffle()

            if self.parameters._configuration["gpu"]:
                torch.cuda.synchronize()
                tsample = time.time()
                t0 = time.time()
                batchid = 0
                for loader in self.training_data_loaders:
                    graph_ions: dgl.DGLGraph
                    graph_grid: dgl.DGLGraph

                    embedding_hash = None
                    embedding_extended = None
                    embedding_step_counter = 0

                    for graph_ions, graph_grid in tqdm(loader, desc="training", disable=self.parameters_full.verbosity < 2):
                        if batchid == self.parameters.profiler_range[0]:
                            torch.cuda.profiler.start()
                        if batchid == self.parameters.profiler_range[1]:
                            torch.cuda.profiler.stop()

                        torch.cuda.nvtx.range_push(f"step {batchid}")

                        graph_ions_hash = hash(graph_ions.edata['rel_pos'][:100].numpy().tobytes())
                        if graph_ions_hash != embedding_hash or\
                            embedding_step_counter >= self.parameters.embedding_reuse_steps:
                            self.encoder_optimizer.step()
                            self.encoder_optimizer.zero_grad()
                            torch.cuda.nvtx.range_push("embedding calcuation")
                            embedding_extended = self._compute_embedding(graph_ions, graph_grid)
                            torch.cuda.nvtx.range_pop()
                            embedding_hash = graph_ions_hash
                            embedding_step_counter = 0
                        embedding_step_counter += 1

                        torch.cuda.nvtx.range_push("data copy in")
                        graph_grid = graph_grid.to(
                            self.parameters._configuration["device"], non_blocking=True
                        )
                        # data copy in
                        torch.cuda.nvtx.range_pop()

                        train_loss = self.__process_mini_batch(
                            self.network, graph_ions, graph_grid, embedding_extended
                        )
                        # step
                        torch.cuda.nvtx.range_pop()
                        loss_float = train_loss.detach().cpu().item()
                        training_loss_sum += loss_float
                        training_loss_sum_logging += loss_float

                        if batchid != 0 and (batchid + 1) % self.parameters.training_report_frequency == 0:
                            torch.cuda.synchronize()
                            sample_time = time.time() - tsample
                            avg_sample_time = sample_time / self.parameters.training_report_frequency
                            avg_sample_tput = self.parameters.training_report_frequency \
                                * (graph_grid.num_nodes()-graph_ions.num_nodes()) / sample_time
                            printout(f"batch {batchid + 1}, "#/{total_samples}, "
                                     f"train avg time: {avg_sample_time} "
                                     f"train avg throughput: {avg_sample_tput}",
                                     min_verbosity=3)
                            tsample = time.time()
                        

                            # summary_writer tensor board
                            if self.parameters.visualisation:
                                training_loss_mean = training_loss_sum_logging / self.parameters.training_report_frequency
                                self.tensor_board.add_scalars(
                                    'Loss', {'training': training_loss_mean}, total_batch_id
                                )
                                self.tensor_board.close()
                                training_loss_sum_logging = 0.0

                        batchid += 1
                        total_batch_id += 1
                torch.cuda.synchronize()
                t1 = time.time()
                printout(f"training time: {t1 - t0}", min_verbosity=2)

                training_loss = training_loss_sum / batchid

                # Calculate the validation loss. and output it.
                torch.cuda.synchronize()
            else:
                batchid = 0

                for loader in self.training_data_loaders:
                    graph_ions: dgl.DGLGraph
                    graph_grid: dgl.DGLGraph
                    embeddings = {}

                    for graph_ions, graph_grid in loader:
                        graph_ions_hash = hash(graph_ions.edata['rel_pos'][:100].numpy().tobytes())
                        if graph_ions_hash not in embeddings:
                            torch.cuda.nvtx.range_push("embedding calcuation")
                            embedding_extended = self._compute_embedding(graph_ions, graph_grid)
                            torch.cuda.nvtx.range_pop()
                            embeddings[graph_ions_hash] = embedding_extended
                        embedding_extended = embeddings[graph_ions_hash]

                        train_loss = self.__process_mini_batch(
                            self.network, graph_ions, graph_grid,
                            embedding_extended
                        ).detach().cpu().item()
                        training_loss_sum += train_loss
                        batchid += 1
                        total_batch_id += 1
                    self.encoder_optimizer.step()
                    self.encoder_optimizer.zero_grad()
                training_loss = training_loss_sum / batchid

            vloss = self.__validate_network(
                self.network, "validation", self.parameters.during_training_metric
            )

            if self.parameters_full.use_horovod:
                vloss = self.__average_validation(vloss, 'average_loss')
            if self.parameters_full.verbosity > 1:
                printout(
                    f"Epoch {epoch}: validation data loss: {vloss}, "
                    f"training data loss: {training_loss}",
                    min_verbosity=2
                )
            else:
                printout(
                    f"Epoch {epoch}: validation data loss: {vloss}",
                    min_verbosity=1
                )

            # summary_writer tensor board
            if self.parameters.visualisation:
                self.tensor_board.add_scalars(
                    'Loss', {'validation': vloss}, total_batch_id
                )
                if self.parameters.visualisation == 2:
                    for name, param in self.network.named_parameters():
                        self.tensor_board.add_histogram(name, param, epoch)
                        self.tensor_board.add_histogram(
                            f'{name}.grad', param.grad, epoch
                        )
                self.tensor_board.close()

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
                if self.parameters.learning_rate_scheduler ==\
                        "ReduceLROnPlateau":
                    self.encoder_scheduler.step(vloss)
            if self.decoder_scheduler is not None:
                if self.parameters.learning_rate_scheduler ==\
                        "ReduceLROnPlateau":
                    self.decoder_scheduler.step(vloss)

            # If early stopping is used, check if we need to do something.
            if self.parameters.early_stopping_epochs > 0:
                if vloss < vloss_old * (1.0 - self.parameters.
                                        early_stopping_threshold):
                    self.patience_counter = 0
                    vloss_old = vloss
                else:
                    self.patience_counter += 1
                    printout("Validation accuracy has not improved "
                             "enough.", min_verbosity=1)
                    if self.patience_counter >= self.parameters.\
                            early_stopping_epochs:
                        printout("Stopping the training, validation "
                                 "accuracy has not improved for",
                                 self.patience_counter,
                                 "epochs.", min_verbosity=1)
                        self.last_epoch = epoch
                        break

            # If checkpointing is enabled, we need to checkpoint.
            if self.parameters.checkpoints_each_epoch != 0:
                checkpoint_counter += 1
                if checkpoint_counter >= \
                        self.parameters.checkpoints_each_epoch:
                    printout("Checkpointing training.", min_verbosity=0)
                    self.last_epoch = epoch
                    self.last_loss = vloss_old
                    self.__create_training_checkpoint()
                    checkpoint_counter = 0

            printout("Time for epoch[s]:", time.time() - start_time,
                     min_verbosity=2)

        ############################
        # CALCULATE FINAL METRICS
        ############################

        if self.parameters.after_before_training_metric != \
                self.parameters.during_training_metric:
            vloss = self.__validate_network(self.network,
                                            "validation",
                                            self.parameters.
                                            after_before_training_metric)
            if self.parameters_full.use_horovod:
                vloss = self.__average_validation(vloss, 'average_loss')

        # Calculate final loss.
        self.final_validation_loss = vloss
        printout("Final validation data loss: ", vloss, min_verbosity=0)

        tloss = float("inf")
        if len(self.data.test_data_sets) > 0:
            tloss = self.__validate_network(
                self.network, "test",
                self.parameters.after_before_training_metric
            )
            if self.parameters_full.use_horovod:
                tloss = self.__average_validation(tloss, 'average_loss')
            printout("Final test data loss: ", tloss, min_verbosity=0)
        self.final_test_loss = tloss

        # Clean-up for pre-fetching lazy loading.
        if self.data.parameters.use_lazy_loading_prefetch:
            self.training_data_loaders.cleanup()
            self.validation_data_loaders.cleanup()
            if len(self.data.test_data_sets) > 0:
                self.test_data_loaders.cleanup()

    def __prepare_to_train(self, optimizer_dict):
        """Prepare everything for training."""
        # Configure keyword arguments for DataSampler.
        kwargs = {'num_workers': self.parameters.num_workers,
                  'pin_memory': False}
        if self.parameters_full.use_gpu:
            kwargs['pin_memory'] = True

        # Read last epoch
        if optimizer_dict is not None: 
            self.last_epoch = optimizer_dict['epoch']+1

        # Scale the learning rate according to horovod.
        if self.parameters_full.use_horovod:
            if hvd.size() > 1 and self.last_epoch == 0:
                printout("Rescaling learning rate because multiple workers are"
                         " used for training.", min_verbosity=1)
                self.parameters.learning_rate = self.parameters.learning_rate \
                    * hvd.size()

        # Choose an optimizer to use.
        if self.parameters.trainingtype == "SGD":
            raise Exception("SGD is not supported.")
            # self.optimizer = optim.SGD(self.network.parameters(),
            #                           lr=self.parameters.learning_rate,
            #                           weight_decay=self.parameters.
            #                           weight_decay)
        elif self.parameters.trainingtype == "Adam":
            # self.optimizer = optim.Adam(self.network.parameters(),
            #                             lr=self.parameters.learning_rate,
            #                             weight_decay=self.parameters.
            #                             weight_decay)
            self.encoder_optimizer = optim.Adam(
                self.network.encoder.parameters(),
                lr=self.parameters.learning_rate_embedding,
                weight_decay=self.parameters.weight_decay
            )
            self.decoder_optimizer = optim.Adam(
                self.network.decoder.parameters(),
                lr=self.parameters.learning_rate,
                weight_decay=self.parameters.weight_decay
            )
        elif self.parameters.trainingtype == "FusedAdam":
            raise Exception("FusedAdam is not supported.")
            # if version.parse(torch.__version__) >= version.parse("1.13.0"):
            #     self.optimizer = optim.Adam(self.network.parameters(),
            #                                lr=self.parameters.learning_rate,
            #                                weight_decay=self.parameters.
            #                                weight_decay, fused=True)
            # else:
            #     raise Exception("Training method requires "
            #                     "at least torch 1.13.0.")
        else:
            raise Exception("Unsupported training method.")

        # Load data from pytorch file.
        if optimizer_dict is not None:
            raise Exception("Loading optimizer state is not supported.")
            self.optimizer.\
                load_state_dict(optimizer_dict['optimizer_state_dict'])
            self.patience_counter = optimizer_dict['early_stopping_counter']
            self.last_loss = optimizer_dict['early_stopping_last_loss']

        if self.parameters_full.use_horovod:
            # scaling the batch size for multiGPU per node
            # self.batch_size= self.batch_size*hvd.local_size()

            compression = hvd.Compression.fp16 if self.parameters_full.\
                running.use_compression else hvd.Compression.none

            # If lazy loading is used we do not shuffle the data points on
            # their own, but rather shuffle them
            # by shuffling the files themselves and then reading file by file
            # per epoch.
            # This shuffling is done in the dataset themselves.
            do_shuffle = self.parameters.use_shuffling_for_samplers
            if self.data.parameters.use_lazy_loading:
                do_shuffle = False

            self.train_sampler = torch.utils.data.\
                distributed.DistributedSampler(self.data.training_data_sets[0],
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=do_shuffle)

            self.validation_sampler = torch.utils.data.\
                distributed.DistributedSampler(self.data.validation_data_sets[0],
                                               num_replicas=hvd.size(),
                                               rank=hvd.rank(),
                                               shuffle=False)

            if self.data.test_data_sets:
                self.test_sampler = torch.utils.data.\
                    distributed.DistributedSampler(self.data.test_data_sets[0],
                                                   num_replicas=hvd.size(),
                                                   rank=hvd.rank(),
                                                   shuffle=False)

            # broadcaste parameters and optimizer state from root device to
            # other devices
            hvd.broadcast_parameters(self.network.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.encoder_optimizer, root_rank=0)
            hvd.broadcast_optimizer_state(self.decoder_optimizer, root_rank=0)

            # Wraps the opimizer for multiGPU operation
            self.encoder_optimizer = hvd.DistributedOptimizer(
                self.encoder_optimizer,
                named_parameters=self.network.encoder.named_parameters(),
                compression=compression, op=hvd.Average
            )
            self.decoder_optimizer = hvd.DistributedOptimizer(
                self.decoder_optimizer,
                named_parameters=self.network.decoder.named_parameters(),
                compression=compression, op=hvd.Average
            )

        # Instantiate the learning rate scheduler, if necessary.
        if self.parameters.learning_rate_scheduler == "ReduceLROnPlateau":
            # raise Exception("ReduceLROnPlateau is not supported.")
            self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.encoder_optimizer,
                patience=self.parameters.learning_rate_patience,
                mode="min", factor=self.parameters.learning_rate_decay,
                verbose=True
            )
            self.decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.decoder_optimizer,
                patience=self.parameters.learning_rate_patience,
                mode="min", factor=self.parameters.learning_rate_decay,
                verbose=True
            )
        elif self.parameters.learning_rate_scheduler is None:
            pass
        else:
            raise Exception("Unsupported learning rate scheduler.")
        if self.encoder_scheduler is not None and optimizer_dict is not None:
            self.encoder_scheduler.\
                load_state_dict(optimizer_dict['lr_encoder_scheduler_state_dict'])
        if self.decoder_scheduler is not None and optimizer_dict is not None:
            self.decoder_scheduler.\
                load_state_dict(optimizer_dict['lr_decoder_scheduler_state_dict'])

        # If lazy loading is used we do not shuffle the data points on their
        # own, but rather shuffle them
        # by shuffling the files themselves and then reading file by file per
        # epoch.
        # This shuffling is done in the dataset themselves.
        do_shuffle = self.parameters.use_shuffling_for_samplers
        if self.data.parameters.use_lazy_loading or self.parameters_full.\
                use_horovod:
            do_shuffle = False

        # Prepare data loaders.(look into mini-batch size)
        if isinstance(self.data.training_data_sets[0], GraphDataset):
            # self.training_data_loaders.append(GraphDataLoader(
            #     self.data.training_data_sets[0], batch_size=self.parameters.mini_batch_size,
            #     sampler=self.train_sampler, **kwargs,
            #     shuffle=do_shuffle
            # ))
            self.training_data_loaders.append(
                self.data.training_data_sets[0]
            )
            # Loader 30 min epoch
            # Raw dataset 20 min epoch
            # ! GraphDataLoader is not advantageous in our case
            # self.training_data_loaders.append(DataLoader(
            #     self.data.training_data_sets[0], batch_size=self.parameters.
            #     mini_batch_size, sampler=self.train_sampler, **kwargs,
            #     shuffle=do_shuffle
            # ))
        elif isinstance(self.data.training_data_sets[0], FastTensorDataset):
            # Not shuffling in loader.
            # I manually shuffle the data set each epoch.
            self.training_data_loaders.append(DataLoader(self.data.training_data_sets[0],
                                                         batch_size=None,
                                                         sampler=self.train_sampler,
                                                         **kwargs,
                                                         shuffle=False))
        else:
            if isinstance(self.data.training_data_sets[0], LazyLoadDatasetSingle):
                self.training_data_loaders = MultiLazyLoadDataLoader(self.data.training_data_sets, **kwargs)
            else:
                self.training_data_loaders.append(DataLoader(self.data.training_data_sets[0],
                                                             batch_size=self.parameters.
                                                             mini_batch_size,
                                                             sampler=self.train_sampler,
                                                             **kwargs,
                                                             shuffle=do_shuffle))
        if isinstance(self.data.validation_data_sets[0], GraphDataset):
            self.validation_data_loaders.append(GraphDataLoader(self.data.validation_data_sets[0],
                                                                batch_size=None,
                                                                sampler=
                                                                self.validation_sampler,
                                                                **kwargs))
        elif isinstance(self.data.validation_data_sets[0], FastTensorDataset):
            self.validation_data_loaders.append(DataLoader(self.data.validation_data_sets[0],
                                                           batch_size=None,
                                                           sampler=
                                                           self.validation_sampler,
                                                           **kwargs))
        else:
            if isinstance(self.data.validation_data_sets[0], LazyLoadDatasetSingle):
                self.validation_data_loaders = MultiLazyLoadDataLoader(self.data.validation_data_sets, **kwargs)
            else:
                self.validation_data_loaders.append(DataLoader(self.data.validation_data_sets[0],
                                                               batch_size=self.parameters.
                                                               mini_batch_size * 1,
                                                               sampler=
                                                               self.validation_sampler,
                                                               **kwargs))
        if self.data.test_data_sets:
            if isinstance(self.data.test_data_sets[0], GraphDataset):
                self.test_data_loaders.append(GraphDataLoader(self.data.test_data_sets[0],
                                                              batch_size=None,
                                                              sampler=self.test_sampler,
                                                              **kwargs))
            elif isinstance(self.data.test_data_sets[0], LazyLoadDatasetSingle):
                self.test_data_loaders = MultiLazyLoadDataLoader(self.data.test_data_sets, **kwargs)
            else:
                self.test_data_loaders.append(DataLoader(self.data.test_data_sets[0],
                                                         batch_size=self.parameters.
                                                         mini_batch_size * 1,
                                                         sampler=self.test_sampler,
                                                         **kwargs))

    def __process_mini_batch(
        self, network, graph_ions: dgl.DGLGraph, graph_grid: dgl.DGLGraph,
        embedding_extended: dict = None
    ):
        """Process a mini batch."""
        if self.parameters._configuration["gpu"]:
            if self.parameters.use_graphs and self.train_graph is None:
                # ! Test this out
                raise Exception("Not tested for now")
                printout("Capturing CUDA graph for training.", min_verbosity=2)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                # Warmup for graphs
                with torch.cuda.stream(s):
                    for i in range(20):
                        self.network.zero_grad(set_to_none=True)

                        with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                            # prediction = network(graph_ions, graph_grid)
                            embedding_extended_ = network.get_embedding(graph_ions, graph_grid)
                            prediction = network.predict_ldos(embedding_extended_, graph_ions, graph_grid)
                            loss = network.calculate_loss(prediction, graph_ions, graph_grid)

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
                    with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                        self.static_prediction = network.predict_ldos(
                            self.static_embedding_extended, self.static_graph_ions, self.static_graph_grid
                        )
                        self.static_loss = network.calculate_loss(
                            self.static_prediction, self.static_graph_ions, self.static_graph_grid
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
                src_ions_static, dst_ions_static = self.static_graph_ions.edges()
                src_ions_static.copy_(src_ions)
                dst_ions_static.copy_(dst_ions)
                for key, value in graph_ions.ndata.items():
                    self.static_graph_ions.ndata[key].copy_(value)
                for key, value in graph_ions.edata.items():
                    self.static_graph_ions.edata[key].copy_(value)
                
                src_grid, dst_grid = graph_grid.edges()
                src_grid_static, dst_grid_static = self.static_graph_grid.edges()
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

                with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                    torch.cuda.nvtx.range_push("forward")
                    prediction = network.predict_ldos(embedding_extended, graph_ions, graph_grid)
                    # forward
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("loss")
                    loss = network.calculate_loss(prediction, graph_ions, graph_grid)
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
            torch.cuda.nvtx.range_pop() # optimizer

            if self.train_graph:
                return self.static_loss
            else:
                return loss
        else:
            raise ValueError("CPU training not supported")
            # prediction = network(graph_ions, graph_grid)
            prediction = network.predict_ldos(embedding_extended, graph_ions, graph_grid)
            loss = network.calculate_loss(prediction, graph_ions, graph_grid)
            loss.backward()
            self.decoder_optimizer.step()
            self.decoder_optimizer.zero_grad()
            return loss

    def __validate_network(self, network, data_set_type, validation_type):
        """Validate a network, using test or validation data."""
        if data_set_type == "test":
            data_loaders = self.test_data_loaders
            data_sets = self.data.test_data_sets
            number_of_snapshots = self.data.nr_test_snapshots
            offset_snapshots = self.data.nr_validation_snapshots + \
                               self.data.nr_training_snapshots

        elif data_set_type == "validation":
            data_loaders = self.validation_data_loaders
            data_sets = self.data.validation_data_sets
            number_of_snapshots = self.data.nr_validation_snapshots
            offset_snapshots = self.data.nr_training_snapshots

        else:
            raise Exception("Please select test or validation"
                            "when using this function.")
        network.eval()
        if validation_type == "ldos":
            # validation_loss_sum = torch.zeros(1, device=self.parameters.
                                            #   _configuration["device"])
            validation_loss_sum = 0
            with torch.no_grad():
                if self.parameters._configuration["gpu"]:
                    report_freq = self.parameters.training_report_frequency
                    torch.cuda.synchronize()
                    tsample = time.time()
                    batchid = 0
                    for loader in data_loaders:
                        printout(f"Validating {validation_type} on {data_set_type} data set.")
                        graph_ions: dgl.DGLGraph
                        graph_grid: dgl.DGLGraph
                        
                        embeddings = {}

                        for graph_ions, graph_grid in tqdm(loader, disable=self.parameters_full.verbosity < 2):
                            graph_grid = graph_grid.to(
                                self.parameters._configuration["device"], non_blocking=True
                            )
                            graph_ions_hash = hash(graph_ions.edata['rel_pos'][:100].numpy().tobytes())
                            if graph_ions_hash not in embeddings:
                                torch.cuda.nvtx.range_push("embedding calcuation")
                                embedding_extended = self._compute_embedding(graph_ions, graph_grid)
                                torch.cuda.nvtx.range_pop()
                                embeddings[graph_ions_hash] = embedding_extended
                            embedding_extended = embeddings[graph_ions_hash]

                            if self.parameters.use_graphs and self.validation_graph is None:
                                printout("Capturing CUDA graph for validation.", min_verbosity=2)
                                s = torch.cuda.Stream()
                                s.wait_stream(torch.cuda.current_stream())
                                # Warmup for graphs
                                with torch.cuda.stream(s):
                                    for _ in range(20):
                                        with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                                            embedding_extended_ = network.get_embedding(graph_ions, graph_grid)
                                            prediction = network.predict_ldos(embedding_extended_, graph_ions, graph_grid)
                                            loss = network.calculate_loss(prediction, graph_ions, graph_grid)
                                torch.cuda.current_stream().wait_stream(s)

                                # Create static entry point tensors to graph

                                self.static_graph_ions_validation = graph_ions.clone()
                                for key, value in graph_ions.ndata.items():
                                    self.static_graph_ions_validation.ndata[key] = value.clone()
                                for key, value in graph_ions.edata.items():
                                    self.static_graph_ions_validation.edata[key] = value.clone()

                                self.static_graph_grid_validation = graph_grid.clone()
                                for key, value in graph_grid.ndata.items():
                                    self.static_graph_grid_validation.ndata[key] = value.clone()
                                for key, value in graph_grid.edata.items():
                                    self.static_graph_grid_validation.edata[key] = value.clone()

                                self.static_embedding_extended_validation = {}
                                for key, value in embedding_extended.items():
                                    self.static_embedding_extended_validation[key] = value.clone()
                                

                                # Capture graph
                                self.validation_graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(self.validation_graph):
                                    with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                                        self.static_prediction_validation = network.predict_ldos(
                                            self.static_embedding_extended_validation,
                                            self.static_graph_ions_validation, self.static_graph_grid_validation,
                                        )
                                        self.static_loss_validation = network.calculate_loss(
                                            self.static_prediction_validation, self.static_graph_ions_validation,
                                            self.static_graph_grid_validation)

                            if self.validation_graph:
                                # ! Assumes same number of nodes and edges

                                # Copy connections (edges) to static graph
                                src_ions, dst_ions = graph_ions.edges()
                                src_ions_static, dst_ions_static = self.static_graph_ions_validation.edges()
                                src_ions_static.copy_(src_ions)
                                dst_ions_static.copy_(dst_ions)
                                src_grid, dst_grid = graph_grid.edges()
                                src_grid_static, dst_grid_static = self.static_graph_grid_validation.edges()
                                src_grid_static.copy_(src_grid)
                                dst_grid_static.copy_(dst_grid)

                                # Copy data to static tensors
                                for key, value in graph_ions.ndata.items():
                                    self.static_graph_ions_validation.ndata[key].copy_(value)
                                for key, value in graph_ions.edata.items():
                                    self.static_graph_ions_validation.edata[key].copy_(value)

                                for key, value in graph_grid.ndata.items():
                                    self.static_graph_grid_validation.ndata[key].copy_(value)
                                for key, value in graph_grid.edata.items():
                                    self.static_graph_grid_validation.edata[key].copy_(value)
                                
                                for key, value in embedding_extended.items():
                                    self.static_embedding_extended_validation[key].copy_(value)

                                self.validation_graph.replay()
                                validation_loss_sum += self.static_loss_validation.detach().cpu().item()
                            else:
                                with torch.cuda.amp.autocast(enabled=self.parameters.use_mixed_precision):
                                    prediction = network.predict_ldos(embedding_extended, graph_ions, graph_grid)
                                    loss = network.calculate_loss(prediction, graph_ions, graph_grid)
                                    validation_loss_sum += loss.detach().cpu().item()
                            if batchid != 0 and (batchid + 1) % report_freq == 0:
                                torch.cuda.synchronize()
                                sample_time = time.time() - tsample
                                avg_sample_time = sample_time / report_freq
                                avg_sample_tput = report_freq * (graph_grid.num_nodes()-graph_ions.num_nodes()) / sample_time
                                printout(f"batch {batchid + 1}, " #/{total_samples}, "
                                         f"validation avg time: {avg_sample_time} "
                                         f"validation avg throughput: {avg_sample_tput}",
                                         min_verbosity=3)
                                tsample = time.time()
                            graph_grid.to('cpu', non_blocking=True)
                            batchid += 1
                    torch.cuda.synchronize()
                else: # CPU
                    raise Exception("CPU validation not supported")
                    batchid = 0
                    for loader in data_loaders:
                        embeddings = {}

                        printout(f"Validating {validation_type} on {data_set_type} data set.")
                        for graph_ions, graph_grid in tqdm(loader, disable=self.parameters_full.verbosity < 2):
                            graph_grid = graph_grid.to(self.parameters._configuration["device"], non_blocking=True)

                            graph_ions_hash = hash(graph_ions.edata['rel_pos'][:100].numpy().tobytes())
                            if graph_ions_hash not in embeddings:
                                torch.cuda.nvtx.range_push("embedding calcuation")
                                embedding_extended = self._compute_embedding(graph_ions, graph_grid)
                                torch.cuda.nvtx.range_pop()
                                embeddings[graph_ions_hash] = embedding_extended
                            embedding_extended = embeddings[graph_ions_hash]

                            prediction = network.predict_ldos(embedding_extended, graph_ions, graph_grid)
                            validation_loss_sum += \
                                network.calculate_loss(prediction, graph_ions, graph_grid).detach().cpu().item()
                            graph_grid.to('cpu', non_blocking=True)
                            batchid += 1

            validation_loss = validation_loss_sum / batchid
            return validation_loss
        elif validation_type == "band_energy" or \
                validation_type == "total_energy":
            errors = []
            if isinstance(self.validation_data_loaders, MultiLazyLoadDataLoader):
                loader_id = 0
                for loader in data_loaders:
                    grid_size = self.data.parameters. \
                        snapshot_directories_list[loader_id +
                                                  offset_snapshots].grid_size

                    actual_outputs = np.zeros(
                        (grid_size, self.data.output_dimension))
                    predicted_outputs = np.zeros(
                        (grid_size, self.data.output_dimension))
                    last_start = 0

                    embeddings = {}

                    printout(f"Validating {validation_type} on {data_set_type} data set.")
                    for graph_ions, graph_grid in tqdm(loader, disable=self.parameters_full.verbosity < 2):
                        graph_grid = graph_grid.to(self.parameters._configuration["device"], non_blocking=True)
                        
                        # TODO: check if this makes sense
                        length = int(graph_grid.number_of_nodes()-graph_ions.number_of_nodes())

                        graph_ions_hash = hash(graph_ions.edata['rel_pos'][:100].numpy().tobytes())
                        if graph_ions_hash not in embeddings:
                            torch.cuda.nvtx.range_push("embedding calcuation")
                            embedding_extended = self._compute_embedding(graph_ions, graph_grid)
                            torch.cuda.nvtx.range_pop()
                            embeddings[graph_ions_hash] = embedding_extended
                        embedding_extended = embeddings[graph_ions_hash]

                        predicted_outputs[last_start:last_start + length, :] = \
                            self.data.output_data_scaler. \
                                inverse_transform(self.network.predict_ldos(embedding_extended, graph_ions, graph_grid).
                                                  to('cpu'), as_numpy=True)
                        actual_outputs[last_start:last_start + length, :] = \
                            self.data.output_data_scaler.inverse_transform(
                                graph_grid.ndata['target'][graph_ions.number_of_nodes():].to('cpu'), as_numpy=True)
                        graph_grid.to('cpu', non_blocking=True)
                        last_start += length
                    errors.append(self._calculate_energy_errors(
                        actual_outputs, predicted_outputs, validation_type,
                        loader_id+offset_snapshots
                    ))
                    loader_id += 1

            else:
                for snapshot_number in range(offset_snapshots,
                                             number_of_snapshots+offset_snapshots):
                    # Get optimal batch size and number of batches per snapshotss
                    grid_size = self.data.parameters.\
                        snapshot_directories_list[snapshot_number].grid_size

                    optimal_batch_size = self._correct_batch_size_for_testing(
                        grid_size, self.parameters.mini_batch_size
                    )
                    number_of_batches_per_snapshot = int(grid_size /
                                                         optimal_batch_size)

                    actual_outputs, predicted_outputs = self._forward_entire_snapshot(
                        snapshot_number, data_sets[0], data_set_type[0:2],
                        number_of_batches_per_snapshot, optimal_batch_size
                    )

                    errors.append(self._calculate_energy_errors(
                        actual_outputs, predicted_outputs, validation_type,
                        snapshot_number
                    ))
            return np.mean(errors)
        else:
            raise Exception("Selected validation method not supported.")

    def _calculate_energy_errors(self, actual_outputs, predicted_outputs,
                                 energy_type, snapshot_number):
        self.data.target_calculator.\
            read_additional_calculation_data(self.data.
                                             get_snapshot_calculation_output(snapshot_number))
        if energy_type == "band_energy":
            try:
                fe_actual = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(actual_outputs)
                be_actual = self.data.target_calculator. \
                    get_band_energy(actual_outputs, fermi_energy=fe_actual)

                fe_predicted = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(predicted_outputs)
                be_predicted = self.data.target_calculator. \
                    get_band_energy(predicted_outputs,
                                    fermi_energy=fe_predicted)
                return np.abs(be_predicted - be_actual) * \
                       (1000 / len(self.data.target_calculator.atoms))
            except ValueError:
                # If the training went badly, it might be that the above
                # code results in an error, due to the LDOS being so wrong
                # that the estimation of the self consistent Fermi energy
                # fails.
                return float("inf")
        elif energy_type == "total_energy":
            try:
                fe_actual = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(actual_outputs)
                be_actual = self.data.target_calculator. \
                    get_total_energy(ldos_data=actual_outputs,
                                     fermi_energy=fe_actual)

                fe_predicted = self.data.target_calculator. \
                    get_self_consistent_fermi_energy(predicted_outputs)
                be_predicted = self.data.target_calculator. \
                    get_total_energy(ldos_data=predicted_outputs,
                                    fermi_energy=fe_predicted)
                return np.abs(be_predicted - be_actual) * \
                       (1000 / len(self.data.target_calculator.atoms))
            except ValueError:
                # If the training went badly, it might be that the above
                # code results in an error, due to the LDOS being so wrong
                # that the estimation of the self consistent Fermi energy
                # fails.
                return float("inf")

        else:
            raise Exception("Invalid energy type requested.")


    def __create_training_checkpoint(self):
        """
        Create a checkpoint during training.

        Follows https://pytorch.org/tutorials/recipes/recipes/saving_and_
        loading_a_general_checkpoint.html to some degree.
        """
        encoder_optimizer_name = self.parameters.checkpoint_name \
            + ".encoder_optimizer.pth"
        decoder_optimizer_name = self.parameters.checkpoint_name \
            + ".decoder_optimizer.pth"

        # Next, we save all the other objects.

        if self.parameters_full.use_horovod:
            if hvd.rank() != 0:
                return
        if self.encoder_scheduler is None and self.decoder_scheduler is None:
            save_dict = {
                'epoch': self.last_epoch,
                'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
                'early_stopping_counter': self.patience_counter,
                'early_stopping_last_loss': self.last_loss
            }
        else:
            save_dict = {
                'epoch': self.last_epoch,
                'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
                'lr_encoder_scheduler_state_dict': self.encoder_scheduler.state_dict(),
                'lr_decoder_scheduler_state_dict': self.decoder_scheduler.state_dict(),
                'early_stopping_counter': self.patience_counter,
                'early_stopping_last_loss': self.last_loss
            }
        torch.save(save_dict, encoder_optimizer_name,
                   _use_new_zipfile_serialization=False)
        torch.save(save_dict, decoder_optimizer_name,
                   _use_new_zipfile_serialization=False)

        self.save_run(self.parameters.checkpoint_name, save_runner=True)

    @staticmethod
    def __average_validation(val, name):
        """Average validation over multiple parallel processes."""
        tensor = torch.tensor(val)
        avg_loss = hvd.allreduce(tensor, name=name, op=hvd.Average)
        return avg_loss.item()
