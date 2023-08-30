# 10 layer  

```python
parameters.data.data_splitting_type = "by_snapshot"
parameters.data.use_graph_data_set = True
parameters.data.n_closest_ions = 8
parameters.data.n_closest_ldos = 16
# parameters.data.n_batches = 6000

parameters.running.max_number_epochs = 100
# len(cartesian_ldos_positions) == 486000
parameters.running.ldos_grid_batch_size = 1000
# 600  -> VRAM 13000MB
# 2000 -> VRAM 52500MB
parameters.running.mini_batch_size = 1
parameters.running.trainingtype = "Adam"
parameters.running.weight_decay = 10**(-7)

parameters.running.run_name = "10layer_wd1e-7_lr5x"

hidden_layers = 10
hidden_layer_size = 128

parameters.running.learning_rate = 5*10**(-6)
parameters.running.learning_rate_embedding = 5*10**(-5)
parameters.running.embedding_reuse_steps = 10
parameters.running.learning_rate_scheduler = 'ReduceLROnPlateau'
parameters.running.learning_rate_decay = 0.1
parameters.running.learning_rate_patience = 0


# n_train = 1
# n_val = 1
# n_test = 1

n_train = 14
n_val = 2
n_test = 4
```