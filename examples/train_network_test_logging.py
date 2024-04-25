import mala
from mala import printout
import os
import argparse

parser = argparse.ArgumentParser(description="Train an MLP on Al data - test logging.")
parser.add_argument('--name', type=str, help="The name of the run.", default='Logging test')
parser.add_argument('--max_number_epochs', type=int, default=10)

parser.add_argument('--hidden_layers', type=int, default=5)
parser.add_argument('--hidden_layer_size', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=10**(-7))
parser.add_argument('--learning_rate', type=float, default=10**(-5))
parser.add_argument('--batch_size', type=int, default=1024)

parser.add_argument('--n_train', type=int, default=4)
parser.add_argument('--n_val', type=int, default=2)
parser.add_argument('--n_test', type=int, default=0)

if __name__ == "__main__":
  args = parser.parse_args()

  print("Arguments used:")
  print(args)

  run_name = args.name
  snap_path = "/bigdata/casus/wdm/Al/933K/2.699gcc/N256/snap/"
  ldos_path = "/bigdata/casus/wdm/Al/933K/2.699gcc/N256/ldos/"
  out_path  = "/bigdata/casus/wdm/Al/933K/2.699gcc/N256/outputs/"

  # Load the parameters from the hyperparameter optimization
  parameters = mala.Parameters()

  # LDOS parameters.
  parameters.targets.ldos_gridsize = 250
  parameters.targets.ldos_gridspacing_ev = 0.1
  parameters.targets.ldos_gridoffset_ev = -10
  parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Al/"
  parameters.network.layer_activations = ["LeakyReLU"]

  # Specify the training parameters, these are not completely the same as
  # for the hyperparameter optimization.
  parameters.running.learning_rate = args.learning_rate
  parameters.running.optimizer = "Adam"
  parameters.running.learning_rate_scheduler = "ReduceLROnPlateau"
  parameters.running.learning_rate_decay = 0.2
  parameters.running.learning_rate_patience = 4
  parameters.running.during_training_metric = "band_energy"
  parameters.verbosity = 2
  parameters.running.logging = 1
  parameters.running.training_log_interval = 1000
  parameters.running.max_number_epochs = args.max_number_epochs
  parameters.running.use_shuffling_for_samplers = False
  parameters.use_gpu = True
  # parameters.running.use_graphs = True # ! doesn't work currently
  parameters.running.weight_decay = args.weight_decay
  parameters.running.validation_metrics = ["lossclear", "fermi_energy", "band_energy", "total_energy", "gradient_magnitude", "gradient_variance"]
  parameters.running.validate_on_training_data = True
  # GNN stuff

  parameters.network.nn_type = "feed-forward"

  parameters.data.use_lazy_loading = False
  parameters.running.learning_rate = args.learning_rate
  parameters.running.mini_batch_size = args.batch_size
  # parameters.running.checkpoints_each_epoch = 1
  parameters.running.checkpoint_best_so_far = True

  parameters.running.run_name = f"\
{run_name}_\
mlp{args.hidden_layers}x{args.hidden_layer_size}_\
lr{parameters.running.learning_rate:.1e}\
"

  parameters.running.checkpoint_name = parameters.running.run_name+"_chkpnt"

  # data_handler = mala.DataHandler(parameters)
  # for i in range(args.n_train):
  #   data_handler.add_snapshot(
  #     f"Al_snapshot{i}.in.npy", snap_path,
  #     f"Al_snapshot{i}.out.npy", ldos_path,
  #     add_snapshot_as="tr",
  #     output_units="1/(Ry*Bohr^3)",
  #     calculation_output_file=os.path.join(out_path, f"Al_snapshot{i}.out")
  #   )
  # for i in range(args.n_train, args.n_train + args.n_val):
  #   data_handler.add_snapshot(
  #     f"Al_snapshot{i}.in.npy", snap_path,
  #     f"Al_snapshot{i}.out.npy", ldos_path,
  #     add_snapshot_as="va",
  #     output_units="1/(Ry*Bohr^3)",
  #     calculation_output_file=os.path.join(out_path, f"Al_snapshot{i}.out")
  #   )
  # data_handler.prepare_data(reparametrize_scaler=False)


  from mala.datahandling.data_repo import data_repo_path
  data_path = os.path.join(data_repo_path, "Be2")
  data_handler = mala.DataHandler(parameters)
  data_handler.add_snapshot(
    "Be_snapshot0.in.npy", data_path,
    "Be_snapshot0.out.npy", data_path, "tr",
    calculation_output_file = os.path.join(data_path, "Be_snapshot0.out")
  )
  data_handler.add_snapshot(
    "Be_snapshot1.in.npy", data_path,
    "Be_snapshot1.out.npy", data_path, "tr",
    calculation_output_file = os.path.join(data_path, "Be_snapshot1.out")
  )
  data_handler.add_snapshot(
    "Be_snapshot2.in.npy", data_path,
    "Be_snapshot2.out.npy", data_path, "va",
    calculation_output_file = os.path.join(data_path, "Be_snapshot2.out")
  )
  data_handler.add_snapshot(
    "Be_snapshot3.in.npy", data_path,
    "Be_snapshot3.out.npy", data_path, "va",
    calculation_output_file = os.path.join(data_path, "Be_snapshot3.out")
  )
  data_handler.prepare_data(reparametrize_scaler=False)

  # Build and train network.
  parameters.network.layer_sizes = [
    data_handler.input_dimension
  ] + args.hidden_layers*[args.hidden_layer_size] + [
    data_handler.output_dimension
  ]
  network = mala.Network(parameters)
  trainer = mala.Trainer(parameters, network, data_handler)

  del data_handler

  try:
    trainer.train_network()
  except KeyboardInterrupt:
    printout("Training was interrupted, saving run.")
  printout("Parameters used:")
  parameters.show()

