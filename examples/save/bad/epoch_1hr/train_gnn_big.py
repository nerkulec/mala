from torch.profiler import profile, record_function, ProfilerActivity
import torch
import total_energy as te
from mala.network import TesterGraph
from mala import printout
import mala
import os
# set CUDA to highest debug mode
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.device_count()

parameters = mala.Parameters()

parameters.data.data_splitting_type = "by_snapshot"
parameters.data.use_graph_data_set = True
parameters.data.n_closest_ions = 8
parameters.data.n_closest_ldos = 16
# parameters.data.n_batches = 6000

parameters.running.max_number_epochs = 200
# len(cartesian_ldos_positions) == 486000
parameters.running.ldos_grid_batch_size = 600
parameters.running.mini_batch_size = 1
parameters.running.learning_rate_embedding = 0.001
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"
parameters.running.visualisation = 1
parameters.running.weight_decay = 0.01

# n_train = 1
# n_val = 1
# n_test = 1

n_train = 14
n_val = 4
n_test = 2

# ! TODO: log magnitude of weights (separately for embedding layers and for the rest)


parameters.targets.ldos_gridsize = 201
parameters.targets.ldos_gridoffset_ev = -13.5
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Bartek_H2/H128"


parameters.verbosity = 2

parameters.use_gpu = True

model_name = "GNN_training_test_big_model"

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
    256,
    256,
    128,
    train_data_handler.output_dimension
]
# Setup network and trainer.
network = mala.Network(parameters)
# network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
trainer = mala.TrainerGraph(parameters, network, train_data_handler)
# try:
trainer.train_network()
print("Training finished!")
# except:
#     print("An exception occurred!")


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



