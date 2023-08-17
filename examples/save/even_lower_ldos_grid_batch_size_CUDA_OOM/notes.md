# Even lower ldos_grid_batch_size - CUDA OOM

```python
parameters.data.data_splitting_type = "by_snapshot"
parameters.data.use_graph_data_set = True
parameters.data.n_closest_ions = 8
parameters.data.n_closest_ldos = 16
parameters.data.n_batches = 6000

parameters.running.max_number_epochs = 200
parameters.running.ldos_grid_batch_size = 50
parameters.running.mini_batch_size = 1
parameters.running.learning_rate_embedding = 0.001
parameters.running.learning_rate = 0.00001
parameters.running.trainingtype = "Adam"
parameters.running.visualisation = 1
parameters.running.weight_decay = 0.01

n_train = 1
n_val = 1
n_test = 1
```

Notes:
- Seems like smaller batch size -> more batches -> more memory
- Important! Memory is allocated during:
```python
loss.backward(retain_graph=True) # line 829 in mala/network/trainer_graph.py
```

