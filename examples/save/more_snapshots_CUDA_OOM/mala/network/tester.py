"""Tester class for testing a network."""
try:
    import horovod.torch as hvd
except ModuleNotFoundError:
    # Warning is thrown by Parameters class
    pass
import numpy as np

from mala.common.parameters import printout
from mala.network.runner import Runner
from mala.network.runner_graph import RunnerGraph
from mala.network.tester_base import TesterBase
from mala.targets.ldos import LDOS
from mala.targets.dos import DOS
from mala.targets.density import Density


class Tester(TesterBase, Runner):
    def __init__(self, params, network, data, observables_to_test=["ldos"],
                 output_format="list"):
        """Initialize the Tester class.
        """
        TesterBase.__init__(self, params, network, data, observables_to_test, output_format)
        Runner.__init__(self, params, network, data)

class TesterGraph(TesterBase, RunnerGraph):
    def __init__(self, params, network, data, observables_to_test=["ldos"],
                 output_format="list"):
        """Initialize the TesterGraph class.
        """
        TesterBase.__init__(self, params, network, data, observables_to_test, output_format)
        RunnerGraph.__init__(self, params, network, data)
        

