# %%
from torch.profiler import profile, record_function, ProfilerActivity
import torch
import total_energy as te
from mala.network import Tester
from mala import printout
import mala
import os

from torch.nn import DataParallel


import signal

def sigusr1_handler(sig, frame):
  print('SIGUSR1 received, raising KeyboardInterrupt')
  raise KeyboardInterrupt

signal.signal(signal.SIGUSR1, sigusr1_handler)


parameters = mala.Parameters()

# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis.
parameters.data.data_splitting_type = "by_snapshot"
# parameters.data.use_graph_data_set = True
# parameters.data.n_closest_ldos = 16

# Specify the training parameters.
parameters.running.max_number_epochs = 100
parameters.running.mini_batch_size = 1024
parameters.running.weight_decay = 1*10**(-7)
parameters.running.learning_rate = 1*10**(-4)
parameters.running.trainingtype = "Adam"

hidden_layers = 3
hidden_layer_size = 128

parameters.running.run_name = f"ff{hidden_layers}x{hidden_layer_size}_wd{parameters.running.weight_decay:.1e}_lr{parameters.running.learning_rate}"

parameters.running.visualisation = 1

parameters.targets.ldos_gridsize = 201
parameters.targets.ldos_gridoffset_ev = -13.5
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Bartek_H2/H128"


parameters.verbosity = 2
parameters.running.visualisation = 1
parameters.running.training_report_frequency = 100

parameters.use_gpu = True



n_train = 14
n_val = 2
n_test = 4


train_data_handler = mala.DataHandler(parameters)
for i in range(n_train):
    train_data_handler.add_snapshot(
        f'H_snapshot{i}.in.npy', f'/bigdata/casus/wdm/Bartek_H2/H128/bispectrum/',
        f'H_snapshot{i}.out.npy', '/bigdata/casus/wdm/Bartek_H2/H128/ldos/',
        'tr', calculation_output_file=f'/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot{i}.out'
    )
for i in range(n_train, n_train + n_val):
    train_data_handler.add_snapshot(
        f'H_snapshot{i}.in.npy', f'/bigdata/casus/wdm/Bartek_H2/H128/bispectrum/',
        f'H_snapshot{i}.out.npy', '/bigdata/casus/wdm/Bartek_H2/H128/ldos/',
        'va', calculation_output_file=f'/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot{i}.out'
    )
train_data_handler.prepare_data(reparametrize_scaler=False)

test_data_handler = mala.DataHandler(parameters)
for i in range(n_train + n_val, n_train + n_val + n_test):
    test_data_handler.add_snapshot(
        f'H_snapshot{i}.in.npy', f'/bigdata/casus/wdm/Bartek_H2/H128/bispectrum/',
        f'H_snapshot{i}.out.npy', '/bigdata/casus/wdm/Bartek_H2/H128/ldos/',
        'te', calculation_output_file=f'/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot{i}.out'
    )
test_data_handler.prepare_data(reparametrize_scaler=False)


parameters.network.nn_type = "feed-forward"
parameters.network.layer_sizes = [
    train_data_handler.input_dimension
] + [hidden_layer_size for _ in range(hidden_layers)] + [
    train_data_handler.output_dimension
]
printout("Parameters used:")
parameters.show()

# Setup network and trainer.
network = mala.Network(parameters)

network = DataParallel(network)

trainer = mala.Trainer(parameters, network, train_data_handler)
try:
    trainer.train_network()
except KeyboardInterrupt:
    print("Training interrupted!")
else:
    print("Training finished!")


additional_calculation_data = '/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot0.out'
trainer.save_run(parameters.running.run_name, additional_calculation_data=additional_calculation_data)


network = network.to('cuda')
observables_to_test = [
    "ldos",
    "band_energy",
    "number_of_electrons"
]

tester = Tester(
    parameters, network, test_data_handler,
    observables_to_test=observables_to_test
)

results = tester.test_all_snapshots()

print(results)

printout("Parameters used:")
parameters.show()
