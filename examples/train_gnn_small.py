from torch.profiler import profile, record_function, ProfilerActivity
import torch
import total_energy as te
from mala.network import TesterGraph
from mala import printout
import mala
import os
# set CUDA to highest debug mode
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import signal

def sigusr1_handler(sig, frame):
  print('SIGUSR1 received, raising KeyboardInterrupt')
  raise KeyboardInterrupt

signal.signal(signal.SIGUSR1, sigusr1_handler)


torch.cuda.device_count()

parameters = mala.Parameters()

parameters.data.data_splitting_type = "by_snapshot"
parameters.data.use_graph_data_set = True
parameters.data.n_closest_ions = 8
parameters.data.n_closest_ldos = 16
# parameters.data.n_batches = 6000

parameters.running.max_number_epochs = 200
# len(cartesian_ldos_positions) == 486000
parameters.running.ldos_grid_batch_size = 3000
parameters.running.mini_batch_size = 1
parameters.running.trainingtype = "Adam"
parameters.running.weight_decay = 0.01

parameters.running.learning_rate = 10**(-5)
parameters.running.learning_rate_embedding = 10**(-3)
parameters.running.learning_rate_scheduler = 'ReduceLROnPlateau'
parameters.running.learning_rate_decay = 0.1
parameters.running.learning_rate_patience = 0


# n_train = 1
# n_val = 1
# n_test = 1

n_train = 14
n_val = 2
n_test = 4

# ! TODO: log magnitude of weights (separately for embedding layers and for the rest)


parameters.targets.ldos_gridsize = 201
parameters.targets.ldos_gridoffset_ev = -13.5
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Bartek_H2/H128"


parameters.verbosity = 2
parameters.running.visualisation = 1
parameters.running.training_report_frequency = 100

parameters.use_gpu = True

model_name = "GNN_training_reuse_embedding"

train_data_handler = mala.DataHandlerGraph(parameters)
for i in range(n_train):
    train_data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in', f'/bigdata/casus/wdm/Bartek_H2/H128/snapshot{i}',
        f'H_snapshot{i}.out.npy', '/bigdata/casus/wdm/Bartek_H2/H128/ldos/',
        'tr', calculation_output_file=f'/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot{i}.out'
    )
for i in range(n_train, n_train + n_val):
    train_data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in', f'/bigdata/casus/wdm/Bartek_H2/H128/snapshot{i}',
        f'H_snapshot{i}.out.npy', '/bigdata/casus/wdm/Bartek_H2/H128/ldos/',
        'va', calculation_output_file=f'/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot{i}.out'
    )
train_data_handler.prepare_data(reparametrize_scaler=False)

test_data_handler = mala.DataHandlerGraph(parameters)
for i in range(n_train + n_val, n_train + n_val + n_test):
    test_data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in', f'/bigdata/casus/wdm/Bartek_H2/H128/snapshot{i}',
        f'H_snapshot{i}.out.npy', '/bigdata/casus/wdm/Bartek_H2/H128/ldos/',
        'te', calculation_output_file=f'/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot{i}.out'
    )

test_data_handler.prepare_data(reparametrize_scaler=False)

parameters.network.nn_type = "se3_transformer"
parameters.network.layer_sizes = [
    train_data_handler.input_dimension,
    128,
    train_data_handler.output_dimension
]
# Setup network and trainer.
network = mala.Network(parameters)
# network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
trainer = mala.TrainerGraph(parameters, network, train_data_handler)
try:
    trainer.train_network()
except KeyboardInterrupt:
    print("Training interrupted!")
else:
    print("Training finished!")


additional_calculation_data = '/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot0.out'
trainer.save_run(model_name, additional_calculation_data=additional_calculation_data)


network = network.to('cuda')
observables_to_test = [
    "ldos",
    "band_energy",
    "band_energy_full",
    "number_of_electrons",
    "total_energy",
    "total_energy_full",
    "density",
    "dos"
]

tester = TesterGraph(parameters, network, test_data_handler,
                     observables_to_test=observables_to_test)

results = tester.test_all_snapshots()

print(results)



