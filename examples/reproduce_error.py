import mala
from mala import printout
from os import path
import argparse

run_name = "repr"

# Load the parameters from the hyperparameter optimization
parameters = mala.Parameters()

# Subset of dataset
parameters.data.n_batches = 100 # ! Only for testing

# LDOS parameters.
parameters.targets.ldos_gridsize = 171
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.ldos_gridoffset_ev = -12
parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Bartek_H2/H256"
parameters.network.layer_activations = ["LeakyReLU"]

# Specify the training parameters, these are not completely the same as
# for the hyperparameter optimization.
parameters.running.learning_rate = 0.0001
parameters.running.trainingtype = "Adam"
parameters.running.learning_rate_scheduler = "ReduceLROnPlateau"
parameters.running.learning_rate_decay = 0.2
parameters.running.learning_rate_patience = 4
parameters.verbosity = 2
parameters.running.visualisation = 1
parameters.running.training_report_frequency = 10
parameters.running.max_number_epochs = 2
parameters.running.use_shuffling_for_samplers = False
parameters.use_gpu = True
# parameters.running.use_graphs = True # ! doesn't work currently
parameters.running.weight_decay = 0.0001

parameters.running.checkpoints_each_epoch = 5
parameters.running.checkpoint_name = run_name+"_chkpnt"
parameters.running.during_training_metric = "total_energy"
parameters.running.after_before_training_metric = "total_energy"
# GNN stuff

parameters.network.nn_type = "se3_transformer"

parameters.data.use_graph_data_set = True
parameters.data.use_lazy_loading = True
parameters.data.retain_graphs = True
parameters.data.n_closest_ions = 4
parameters.data.n_closest_ldos = 8

parameters.network.num_heads = 4
parameters.network.channels_div = 2
parameters.network.max_degree = 2

parameters.running.ldos_grid_batch_size = 2000
parameters.running.embedding_reuse_steps = 10
parameters.running.learning_rate_embedding = 0.0001
parameters.running.learning_rate = 0.001
parameters.running.mini_batch_size = 1



parameters.running.run_name = f"\
Al_{run_name}_\
gnn{2}x{16}_\
wd{parameters.running.weight_decay:.1e}_\
lre{parameters.running.learning_rate_embedding:.1e}_\
lr{parameters.running.learning_rate:.1e}_\
nclosest{parameters.data.n_closest_ions}x{parameters.data.n_closest_ldos}_\
nheads{parameters.network.num_heads}_\
reuse{parameters.running.embedding_reuse_steps}_\
"

train_data_handler = mala.DataHandler(parameters)
test_data_handler = mala.DataHandler(parameters)
for pressure in [40, 80, 120, 160][:1]:
  for temperature in [700, 800, 900, 1000][:1]:
    ldos_folder = f'/bigdata/casus/wdm/Bartek_H2/H256/ldos/{pressure}GPa/{temperature}K'
    snapshots_folder = f'/bigdata/casus/wdm/Bartek_H2/H256/snapshots/{pressure}GPa/{temperature}K'
    for i in range(0, 1):
      snapshot_folder = f'{snapshots_folder}/snapshot{i}'
      train_data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in', snapshot_folder,
        f'H_snapshot{i}.out.npy', ldos_folder,
        'tr', calculation_output_file=f'{snapshot_folder}/snapshot{i}.out'
      )
    for i in range(1, 1 + 1):
      snapshot_folder = f'{snapshots_folder}/snapshot{i}'
      train_data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in', snapshot_folder,
        f'H_snapshot{i}.out.npy', ldos_folder,
        'va', calculation_output_file=f'{snapshot_folder}/snapshot{i}.out'
      )
    for i in range(1 + 1, 1 + 1 + 1):
      snapshot_folder = f'{snapshots_folder}/snapshot{i}'
      test_data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in', snapshot_folder,
        f'H_snapshot{i}.out.npy', ldos_folder,
        'te', calculation_output_file=f'{snapshot_folder}/snapshot{i}.out'
      )
train_data_handler.prepare_data(reparametrize_scaler=False)
test_data_handler.prepare_data(reparametrize_scaler=False)

# Build and train network.
parameters.network.layer_sizes = [
    train_data_handler.input_dimension
] + args.hidden_layers*[args.hidden_layer_size] + [
    train_data_handler.output_dimension
]
network = mala.Network(parameters)
trainer = mala.Trainer(parameters, network, train_data_handler)

observables_to_test = [
  "ldos",
  "band_energy",
  "total_energy",
  "number_of_electrons",
  "density",
  "dos"
]
tester = mala.Tester(parameters, network, test_data_handler, observables_to_test=observables_to_test)

try:
    trainer.train_network()
except KeyboardInterrupt:
    printout("Training was interrupted, saving run.")

additional_calculation_data = '/bigdata/casus/wdm/Bartek_H2/H256/snapshots/40GPa/700K/snapshot0/snapshot0.out'
trainer.save_run(parameters.running.run_name, additional_calculation_data=additional_calculation_data)



# ------------
# Test network


results = tester.test_all_snapshots()

printout("Results")
print(results)

printout("Parameters used:")
parameters.show()


