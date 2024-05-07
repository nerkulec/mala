import os
import torch
import mala
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Run the MALA training on multiple GPUs.')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
args = parser.parse_args()

ldos_path = "/bigdata/casus/wdm/Al/933K/2.699gcc/N256/ldos/"
bisp_path  = "/bigdata/casus/wdm/Al/933K/2.699gcc/N256/snap/"
out_path  = "/bigdata/casus/wdm/Al/933K/2.699gcc/N256/output/"

parameters = mala.Parameters()
parameters.data.input_rescaling_type = "feature-wise-standard"
parameters.data.output_rescaling_type = "normal"
parameters.running.max_number_epochs = 10
parameters.running.mini_batch_size = 1024
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"

parameters.use_gpu = args.gpu
parameters.targets.target_type = "LDOS"
parameters.targets.ldos_gridsize = 250
parameters.targets.ldos_gridspacing_ev = 0.1
parameters.targets.ldos_gridoffset_ev = -10
parameters.targets.pseudopotential_path = "/bigdata/casus/wdm/Al/"
parameters.network.layer_activations = ["LeakyReLU"]

n_train = 4
n_val = 1

data_handler = mala.DataHandler(parameters)
for i in range(n_train):
  data_handler.add_snapshot(
    f"Al_snapshot{i}.in.npy", bisp_path,
    f"Al_snapshot{i}.out.npy", ldos_path,
    add_snapshot_as="tr",
    output_units="1/(Ry*Bohr^3)",
    calculation_output_file=os.path.join(out_path, f"Al_snapshot{i}.out")
  )
for i in range(n_train, n_train + n_val):
  data_handler.add_snapshot(
    f"Al_snapshot{i}.in.npy", bisp_path,
    f"Al_snapshot{i}.out.npy", ldos_path,
    add_snapshot_as="va",
    output_units="1/(Ry*Bohr^3)",
    calculation_output_file=os.path.join(out_path, f"Al_snapshot{i}.out")
  )
data_handler.prepare_data()

parameters.network.layer_sizes = \
  [data_handler.input_dimension] \
  + [256]*15 \
  + [data_handler.output_dimension]
network = mala.Network(parameters)

# network = torch.nn.DataParallel(network, device_ids=list(range(args.gpu)))

trainer = mala.Trainer(parameters, network, data_handler)
trainer.train_network()

