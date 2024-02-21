import os

import mala
from mala import printout

"""
ex19_train_gnn.py: Shows how a gnn can be trained on material
data using this framework.
"""


####################
# PARAMETERS
# All parameters are handled from a central parameters class that
# contains subclasses.
####################

parameters = mala.Parameters()

# Specify the training parameters.
parameters.targets.ldos_gridsize = 171

parameters.running.max_number_epochs = 10
parameters.running.ldos_grid_batch_size = 1000
parameters.running.learning_rate = 10**-5
parameters.running.trainingtype = "Adam"
parameters.verbosity = 1
parameters.use_gpu = True

parameters.network.nn_type = "se3_transformer"
parameters.data.use_graph_data_set = True

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandler(parameters)

for i in range(1):
    data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in',
        f'/bigdata/casus/wdm/Bartek_H2/H256/snapshots/40GPa/700K/snapshot{i}',
        f'H_snapshot{i}.out.npy',
        f'/bigdata/casus/wdm/Bartek_H2/H256/ldos/40GPa/700K',
        'tr'
    )
for i in range(1):
    data_handler.add_snapshot(
        f'H_snapshot{i}.pw.scf.in',
        f'/bigdata/casus/wdm/Bartek_H2/H256/snapshots/40GPa/700K/snapshot{i}',
        f'H_snapshot{i}.out.npy',
        f'/bigdata/casus/wdm/Bartek_H2/H256/ldos/40GPa/700K',
        'va'
    )
# for i in range(1):
#     data_handler.add_snapshot(
#         f'H_snapshot{i}.pw.scf.in',
#         f'/bigdata/casus/wdm/Bartek_H2/H256/snapshots/40GPa/700K/snapshot{i}',
#         f'H_snapshot{i}.out.npy',
#         f'/bigdata/casus/wdm/Bartek_H2/H256/ldos/40GPa/700K',
#         'te'
#     )

data_handler.prepare_data()
printout("Read data: DONE.")

####################
# NETWORK SETUP
# Set up the network and trainer we want to use.
# The layer sizes can be specified before reading data,
# but it is safer this way.
####################

parameters.network.layer_sizes = [
    data_handler.input_dimension,
    64,
    data_handler.output_dimension
]

# Setup network and trainer.
network = mala.Network(parameters)
trainer = mala.Trainer(parameters, network, data_handler)
printout("Network setup: DONE.")

####################
# TRAINING
# Train the network and save it afterwards.
####################

printout("Starting training.")
trainer.train_network()

# Additional calculation data, i.e., a simulation output, may be saved
# along side the model. This makes future inference easier.
additional_calculation_data = "/bigdata/casus/wdm/Bartek_H2/H256/snapshots/40GPa/700K/snapshot0/snapshot0.out"
trainer.save_run(
    "H256_model",
    additional_calculation_data=additional_calculation_data
)
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
parameters.show()
