# %%
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

# %%
parameters = mala.Parameters()

# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis.
parameters.data.data_splitting_type = "by_snapshot"
parameters.data.use_graph_data_set = True
parameters.data.n_closest_ldos = 16

# Specify the training parameters.
parameters.running.max_number_epochs = 1
parameters.running.ldos_grid_batch_size = 30
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"
parameters.running.visualisation = 1


parameters.targets.ldos_gridsize = 201
parameters.targets.ldos_gridoffset_ev = -13.5
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Bartek_H2/H128"

parameters.verbosity = 1

parameters.use_gpu = True


# %%
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True, profile_memory=True, with_stack=True,
#     on_trace_ready=torch.profiler.tensorboard_trace_handler("./cuda_profile_tensorboard_logs_data_loading"),
# ) as prof:
#     with record_function("data_loading"):

n_train = 1
n_val = 1
n_test = 1

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


# %%
parameters.network.nn_type = "se3_transformer"
parameters.network.layer_sizes = [
    train_data_handler.input_dimension,
    32,
    train_data_handler.output_dimension
]
# Setup network and trainer.
network = mala.Network(parameters)
trainer = mala.TrainerGraph(parameters, network, train_data_handler)

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True, profile_memory=True, with_stack=True,
#     on_trace_ready=torch.profiler.tensorboard_trace_handler("./cuda_profile_tensorboard_logs"),
# ) as prof:
#     with record_function("model_inference"):

trainer.train_network()
print("Training finished!")

additional_calculation_data = '/bigdata/casus/wdm/Bartek_H2/H128/outputs/snapshot0.out'
trainer.save_run("H2_model_test_addcalc", additional_calculation_data=additional_calculation_data)


# %%
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



