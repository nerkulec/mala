"""Neural network for MALA."""

from abc import abstractmethod
from typing import List
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional

from mala.common.parameters import Parameters
from mala.common.parallelizer import printout


from dgl import DGLGraph
import dgl
from torch import tensor

from se3_transformer.model.basis import get_basis, update_basis_with_fused
from se3_transformer.model.layers.attention import AttentionBlockSE3

from se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.model.layers.norm import NormSE3
from se3_transformer.model.layers.pooling import GPooling
from se3_transformer.runtime.utils import str2bool
from se3_transformer.model.fiber import Fiber
from se3_transformer.model.transformer import (
    Sequential,
    get_populated_edge_features,
)


class Network(nn.Module):
    """
    Central network class for this framework, based on pytorch.nn.Module.

    The correct type of neural network will automatically be instantiated
    by this class if possible. You can also instantiate the desired
    network directly by calling upon the subclass.

    Parameters
    ----------
    params : mala.common.parametes.Parameters
        Parameters used to create this neural network.
    """

    def __new__(cls, params: Parameters = None):
        """
        Create a neural network instance.

        The correct type of neural network will automatically be instantiated
        by this class if possible. You can also instantiate the desired
        network directly by calling upon the subclass.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters used to create this neural network.
        """
        model = None

        # Check if we're accessing through base class.
        # If not, we need to return the correct object directly.
        if cls == Network:
            if params.network.nn_type == "feed-forward":
                model = super(Network, FeedForwardNet).__new__(FeedForwardNet)

            elif params.network.nn_type == "se3_transformer":
                model = super(Network, SE3Transformer).__new__(SE3Transformer)

            if model is None:
                raise Exception("Unsupported network architecture.")
        else:
            model = super(Network, cls).__new__(cls)

        return model

    def __init__(self, params: Parameters):
        # copy the network params from the input parameter object
        self.use_ddp = params.use_ddp
        self.mini_batch_size = params.running.mini_batch_size
        self.params = params.network
        self.parameters_full = params

        # if the user has planted a seed (for comparibility purposes) we
        # should use it.
        if params.manual_seed is not None:
            torch.manual_seed(params.manual_seed)
            torch.cuda.manual_seed(params.manual_seed)

        # initialize the parent class
        super(Network, self).__init__()

        # Mappings for parsing of the activation layers.
        self.activation_mappings = {
            "Sigmoid": nn.Sigmoid,
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "Tanh": nn.Tanh,
        }

        # initialize the layers
        self.number_of_layers = len(self.params.layer_sizes) - 1

        # initialize the loss function
        if self.params.loss_function_type == "mse":
            self.loss_func = functional.mse_loss
        else:
            raise Exception("Unsupported loss function.")

    @abstractmethod
    def forward(self, inputs):
        """Abstract method. To be implemented by the derived class."""
        pass

    def do_prediction(self, array):
        """
        Predict the output values for an input array..

        Interface to do predictions. The data put in here is assumed to be a
        scaled torch.Tensor and in the right units. Be aware that this will
        pass the entire array through the network, which might be very
        demanding in terms of RAM.

        Parameters
        ----------
        array : torch.Tensor
            Input array for which the prediction is to be performed.

        Returns
        -------
        predicted_array : torch.Tensor
            Predicted outputs of array.

        """
        self.eval()
        with torch.no_grad():
            return self(array)

    def calculate_loss(self, output, target):
        """
        Calculate the loss for a predicted output and target.

        Parameters
        ----------
        output : torch.Tensor
            Predicted output.

        target : torch.Tensor.
            Actual output.

        Returns
        -------
        loss_val : float
            Loss value for output and target.

        """
        loss = self.loss_func(output, target)
        if self.parameters_full.running.l1_regularization > 0:
            loss += self.parameters_full.running.l1_regularization * sum(
                [torch.sum(torch.abs(param)) for param in self.parameters()]
            )
        return loss

    # FIXME: This guarentees downwards compatibility, but it is ugly.
    #  Rather enforce the right package versions in the repo.
    def save_network(self, path_to_file):
        """
        Save the network.

        This function serves as an interfaces to pytorchs own saving
        functionalities AND possibly own saving needs.

        Parameters
        ----------
        path_to_file : string
            Path to the file in which the network should be saved.
        """
        # If we use ddp, only save the network on root.
        if self.use_ddp:
            if dist.get_rank() != 0:
                return
        torch.save(
            self.state_dict(),
            path_to_file,
            _use_new_zipfile_serialization=False,
        )

    @classmethod
    def load_from_file(cls, params, file):
        """
        Load a network from a file.

        Parameters
        ----------
        params : mala.common.parameters.Parameters
            Parameters object with which the network should be created.
            Has to be compatible to the network architecture. This is usually
            enforced by using the same Parameters object (and saving/loading
            it to)

        file : string or ZipExtFile
            Path to the file from which the network should be loaded.

        Returns
        -------
        loaded_network : Network
            The network that was loaded from the file.
        """
        loaded_network = Network(params)
        loaded_network.load_state_dict(
            torch.load(file, map_location=params.device)
        )
        loaded_network.eval()
        return loaded_network


class FeedForwardNet(Network):
    """Initialize this network as a feed-forward network."""

    def __new__(cls):
        return super(Network, cls).__new__(cls)

    def __init__(self, params):
        super(FeedForwardNet, self).__init__(params)

        self.layers = nn.ModuleList()

        # If we have only one entry in the activation list,
        # we use it for the entire list.
        # We should NOT modify the list itself. This would break the
        # hyperparameter algorithms.
        use_only_one_activation_type = False
        if len(self.params.layer_activations) == 1:
            use_only_one_activation_type = True
        elif len(self.params.layer_activations) < self.number_of_layers:
            raise Exception("Not enough activation layers provided.")
        elif len(self.params.layer_activations) > self.number_of_layers:
            printout(
                "Too many activation layers provided. The last",
                str(
                    len(self.params.layer_activations) - self.number_of_layers
                ),
                "activation function(s) will be ignored.",
                min_verbosity=1,
            )

        # Add the layers.
        # As this is a feedforward layer we always add linear layers, and then
        # an activation function
        for i in range(0, self.number_of_layers):
            self.layers.append(
                (
                    nn.Linear(
                        self.params.layer_sizes[i],
                        self.params.layer_sizes[i + 1],
                    )
                )
            )
            try:
                if use_only_one_activation_type:
                    self.layers.append(
                        self.activation_mappings[
                            self.params.layer_activations[0]
                        ]()
                    )
                else:
                    self.layers.append(
                        self.activation_mappings[
                            self.params.layer_activations[i]
                        ]()
                    )
            except KeyError:
                raise Exception("Invalid activation type seleceted.")

        # Once everything is done, we can move the Network on the target
        # device.
        self.to(self.params._configuration["device"])

    def forward(self, inputs):
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        inputs : torch.Tensor
            Input array for which the forward pass is to be performed.

        Returns
        -------
        predicted_array : torch.Tensor
            Predicted outputs of array.
        """
        # Forward propagate data.
        if self.parameters_full.running.input_noise > 0 and self.training:
            inputs = inputs + torch.normal(
                mean=0.0,
                std=self.parameters_full.running.input_noise,
                size=inputs.size(),
                device=inputs.device,
            )
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class SE3Encoder(Network):
    """Initialize this network as a SE(3)-Equivariant encoder graph neural network."""

    def __init__(self, params):
        super(SE3Encoder, self).__init__(params)
        self.hidden_size = params.network.layer_sizes[1]

        input_fiber = Fiber({"0": 1})
        hidden_fiber = Fiber(
            {
                str(i): self.hidden_size
                for i in range(self.params.max_degree + 1)
            }
        )
        edge_fiber = Fiber({})

        self.input_layer = AttentionBlockSE3(
            fiber_in=input_fiber,
            fiber_out=hidden_fiber,
            fiber_edge=edge_fiber,
            num_heads=self.params.num_heads,
            channels_div=self.params.channels_div,
            max_degree=self.params.max_degree,
            fuse_level=ConvSE3FuseLevel.FULL,
            low_memory=False,
        )
        self.hidden_layers_ = []
        for _ in range(len(self.params.layer_sizes) - 2):
            self.hidden_layers_.append(
                AttentionBlockSE3(
                    fiber_in=hidden_fiber,
                    fiber_out=hidden_fiber,
                    fiber_edge=edge_fiber,
                    num_heads=self.params.num_heads,
                    channels_div=self.params.channels_div,
                    max_degree=self.params.max_degree,  # ! REDO
                    fuse_level=ConvSE3FuseLevel.FULL,
                    low_memory=False,
                    use_batch_norm=self.parameters_full.running.batch_norm,
                    dropout=self.parameters_full.running.dropout,
                )
            )
        self.hidden_layers = nn.ModuleList(self.hidden_layers_)
        # self.to(self.params._configuration["device"])

    def embed(self, graph_ions: DGLGraph):
        basis_ions = {}
        for key, value in graph_ions.edata.items():
            if key[:6] == "basis_":
                basis_ions[key[6:]] = value
        if self.parameters_full.running.input_noise > 0 and self.training:
            graph_embedding = self.input_layer(
                {
                    "0": graph_ions.ndata["feature"]
                    + torch.normal(
                        mean=0.0,
                        std=self.parameters_full.running.input_noise,
                        size=graph_ions.ndata["feature"].size(),
                        device=graph_ions.ndata["feature"].device,
                    )
                },
                {
                    "0": graph_ions.edata["edge_features"]
                    + torch.normal(
                        mean=0.0,
                        std=self.parameters_full.running.input_noise,
                        size=graph_ions.edata["edge_features"].size(),
                        device=graph_ions.edata["edge_features"].device,
                    )
                },
                graph=graph_ions,
                basis=basis_ions,
            )
        else:
            graph_embedding = self.input_layer(
                {"0": graph_ions.ndata["feature"]},
                {"0": graph_ions.edata["edge_features"]},
                graph=graph_ions,
                basis=basis_ions,
            )
        graph_embedding_local = graph_embedding
        for layer in self.hidden_layers:
            graph_embedding_local = layer(
                graph_embedding_local,
                {"0": graph_ions.edata["edge_features"]},
                graph=graph_ions,
                basis=basis_ions,
            )
        return graph_embedding_local

    def extend_embedding(
        self, graph_embedding: dict, graph_ions: DGLGraph, graph_grid: DGLGraph
    ):
        n_grid = graph_grid.number_of_nodes() - graph_ions.number_of_nodes()
        graph_embedding_extended = {
            str(i): torch.cat(
                [
                    graph_embedding[str(i)],
                    torch.zeros(
                        (n_grid, self.hidden_size, 2 * i + 1),
                        dtype=torch.float32,
                        device=graph_ions.device,
                    ),
                ],
                dim=0,
            )
            for i in range(self.params.max_degree + 1)
        }
        return graph_embedding_extended

    def forward(self, graph_ions: DGLGraph, graph_grid: DGLGraph):
        graph_embedding = self.embed(graph_ions)
        graph_embedding_extended = self.extend_embedding(
            graph_embedding, graph_ions, graph_grid
        )
        return graph_embedding_extended


class SE3Decoder(nn.Module):
    """Initialize this network as a SE(3)-Equivariant decoder graph neural network."""

    def __init__(self, params=None):
        super(SE3Decoder, self).__init__()
        if params is not None:
            # super(SE3Decoder, self).__init__(params)
            self.params = params.network
            self.parameters_full = params
            self.loss_func = functional.mse_loss

            self.hidden_size = params.network.layer_sizes[1]
            self.ldos_size = params.targets.ldos_gridsize

            # hidden_fiber = Fiber({'0': self.hidden_size,  '1': self.hidden_size})
            hidden_fiber = Fiber(
                {
                    str(i): self.hidden_size
                    for i in range(self.params.max_degree + 1)
                }
            )
            ldos_fiber = Fiber({"0": self.ldos_size})
            edge_fiber = Fiber({})

            self.output_layer_grid = AttentionBlockSE3(
                fiber_in=hidden_fiber,
                fiber_out=ldos_fiber,
                fiber_edge=edge_fiber,
                num_heads=1,  # Output layer has to have 1 head
                channels_div=1,
                max_degree=1,
                fuse_level=ConvSE3FuseLevel.FULL,
                low_memory=False,
            )
            # self.to(self.params._configuration["device"])

    def predict_ldos(
        self,
        graph_embedding_extended: dict,
        graph_ions: DGLGraph,
        graph_grid: DGLGraph,
    ):
        basis_grid = {}
        for key, value in graph_grid.edata.items():
            if key[:6] == "basis_":
                basis_grid[key[6:]] = value

        ldos_pred = self.output_layer_grid(
            graph_embedding_extended,
            {"0": graph_grid.edata["edge_features"]},
            graph=graph_grid,
            basis=basis_grid,
        )
        return ldos_pred["0"].squeeze(-1)[graph_ions.number_of_nodes() :]

    def forward(
        self,
        graph_embedding_extended: dict,
        graph_ions: DGLGraph,
        graph_grid: DGLGraph,
    ):
        ldos_pred = self.predict_ldos(
            graph_embedding_extended, graph_ions, graph_grid
        )
        return ldos_pred

    def calculate_loss(self, prediction, graph_ions, graph_grid):
        """
        Calculate the loss on grid values

        Parameters
        ----------
        prediction : torch.Tensor
            Graph containing the ions.

        graph_grid : dgl.Graph
            Graph connecting ions to grid positions

        Returns
        -------
        loss_val : float
            Loss value for prediction and target.

        """
        loss = self.loss_func(
            prediction, graph_grid.ndata["target"][-prediction.shape[0] :]
        )
        # loss_ions = self.loss_func(prediction, graph_ions.ndata['target'])

        if self.parameters_full.running.l1_regularization > 0:
            loss += self.parameters_full.running.l1_regularization * sum(
                [torch.sum(torch.abs(param)) for param in self.parameters()]
            )
        loss_total = loss  # + loss_ions
        return loss_total


class SE3Transformer(Network):
    """Initialize this network as a SE(3)-Equivariant transformer graph neural network."""

    def __init__(self, params):
        super(SE3Transformer, self).__init__(params)
        self.encoder = SE3Encoder(params)
        # self.decoder = torch.nn.DataParallel(SE3Decoder(params))
        self.decoder = SE3Decoder(params)
        self.to(self.params._configuration["device"])

    def get_embedding(self, graph_ions: DGLGraph, graph_grid: DGLGraph):
        graph_embedding_extended = self.encoder(graph_ions, graph_grid)
        return graph_embedding_extended

    def predict_ldos(
        self,
        graph_embedding_extended: dict,
        graph_ions: DGLGraph,
        graph_grid: DGLGraph,
    ):
        ldos_pred = self.decoder(
            graph_embedding_extended, graph_ions, graph_grid
        )
        return ldos_pred

    def forward(self, graph_ions: DGLGraph, graph_grid: DGLGraph):
        graph_embedding = self.embed(graph_ions)
        graph_embedding_extended = self.extend_embedding(
            graph_embedding, graph_ions, graph_grid
        )
        ldos_pred = self.predict_ldos(
            graph_embedding_extended, graph_ions, graph_grid
        )
        return ldos_pred

    def calculate_loss(self, prediction, graph_ions, graph_grid):
        """
        Calculate the loss on grid values

        Parameters
        ----------
        prediction : torch.Tensor
            Graph containing the ions.

        graph_grid : dgl.Graph
            Graph connecting ions to grid positions

        Returns
        -------
        loss_val : float
            Loss value for prediction and target.

        """
        loss = self.decoder.calculate_loss(prediction, graph_ions, graph_grid)
        return loss
