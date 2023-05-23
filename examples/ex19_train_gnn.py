import os

import mala
from mala import printout

from mala.datahandling.data_repo import data_repo_path
data_path = os.path.join(data_repo_path, "Be2")

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

# Currently, the splitting in training, validation and test set are
# done on a "by snapshot" basis.
parameters.data.data_splitting_type = "by_snapshot"

# Specify the training parameters.
parameters.running.max_number_epochs = 100
parameters.running.ldos_grid_batch_size = 40
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"
parameters.verbosity = 1

####################
# DATA
# Add and prepare snapshots for training.
####################

data_handler = mala.DataHandlerGraph(parameters)

for i in range(16):
    data_handler.add_raw_snapshot(
        f'/bigdata/casus/wdm/Bartek_H2/H128/snapshot{i}/H_snapshot{i}.pw.scf.in',
        f'/bigdata/casus/wdm/Bartek_H2/H128/ldos/H_snapshot{i}.out.npy',
        (90, 90, 60, 201), 'tr'
    )
for i in range(16, 20):
    data_handler.add_raw_snapshot(
        f'/bigdata/casus/wdm/Bartek_H2/H128/snapshot{i}/H_snapshot{i}.pw.scf.in',
        f'/bigdata/casus/wdm/Bartek_H2/H128/ldos/H_snapshot{i}.out.npy',
        (90, 90, 60, 201), 'va'
    )

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
additional_calculation_data = os.path.join(data_path, "Be_snapshot0.out")
trainer.save_run("be_model",
                      additional_calculation_data=additional_calculation_data)
printout("Training: DONE.")

####################
# RESULTS.
# Print the used parameters and check whether the loss decreased enough.
####################

printout("Parameters used for this experiment:")
parameters.show()
