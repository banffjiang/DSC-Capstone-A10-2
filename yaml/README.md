# Note:
* Most of these files will not work since they are connected to our wandb work space, you need to edit them
* Also you need to run the following command with your wandb key:
```
kubectl create secret generic wandb-api-key \
  --from-literal=WANDB_API_KEY=<your_actual_key_here> \
  -n <namespace>
```


## Quick description of yaml files

1. `copy_chekpoint.yaml`
- Is a job that duplicates the model checkpoints from one pvc to another, helpful for creating concurrent runs
2. `debug_pod.yaml`
- Job that creates a pod so that you can `exec` into it and run code/check the files
3. `dpo_posttraining_job.yaml`
- Job that runs the dpo fine tuning that uses the checkpoint from SFT and pretraining, is the main job of our project
4. `dpo_pvc.yaml`
- Creates the pvc that we used for the main model training
5. `dpo_sweep_pvc.yaml`
- Creates the pvc that we used for running hyperparameter sweeps
6. `pretraining_job.yaml`
- Runs the pretraining and SFT parts and saves checkpoints that we feed into the `dpo_posttraining_job.yaml`
7. `sweep_job.yaml`
- Runs the hyperparameter sweep, need to change to match the sweep for it to run
8. `/deprecated_yaml_jobs`
- Has many yaml jobs that we used throughout the project but no longer work due to files being changed
