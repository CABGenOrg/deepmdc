import hydra
import torch
import numpy as np
from os import path, environ
from omegaconf import DictConfig
from deeprc.training import train, evaluate
from src.utils.create_orf_table import create_orf_table
from src.utils.handle_files import check_database_presence
from src.utils.handle_machine_learning import build_model, clear_gpu_memory
from deeprc.task_definitions import TaskDefinition, MulticlassTarget, \
    BinaryTarget, RegressionTarget
from deeprc.dataset_readers import make_dataloaders_stratified, \
    no_sequence_count_scaling

environ["PYTORCH_CUDA_ALLOC_CONF"] = ("garbage_collection_threshold:0.8,"
                                      "max_split_size_mb:128,"
                                      "expandable_segments:True")
environ["HYDRA_FULL_ERROR"] = "1"


def create_task_definition(task_config):
    targets = []
    target_cfg = task_config.target

    if target_cfg.type == "binary":
        targets.append(BinaryTarget(
            # Column name of task in metadata file
            column_name=target_cfg.column_name,
            # Entries with value '+' will be positive class, others will
            # be negative class
            true_class_value=target_cfg.positive_class,
            # We can up- or down-weight the positive class if the classes
            # are imbalanced
            pos_weight=target_cfg.pos_weight,
            # Weight of this task for the total training loss
            task_weight=target_cfg.task_weight
        ))
    elif target_cfg.type == "multiclass":
        targets.append(MulticlassTarget(
            # Column name of task in metadata file
            column_name=target_cfg.column_name,
            # Values in task column to expect
            possible_target_values=target_cfg.possible_target_values,
            # Weight individual classes (e.g. if class "S" is
            # overrepresented)
            class_weights=target_cfg.class_weights,
            # Weight of this task for the total training loss
            task_weight=target_cfg.task_weight
        ))
    elif target_cfg.type == "regression":
        targets.append(RegressionTarget(
            # Column name of task in metadata file
            column_name=target_cfg.column_name,
            # Normalize targets by ((target_value - mean) / std)
            normalization_mean=target_cfg.normalization_mean,
            normalization_std=target_cfg.normalization_std,
            # Weight of this task for the total training loss
            task_weight=target_cfg.task_weight
        ))
    else:
        raise ValueError(f"Unsupported task type: {target_cfg.type}")

    return TaskDefinition(targets=targets)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Number of updates to train for.
    # Recommended: int(1e5). Default: int(1e3).
    n_updates = cfg.training.n_updates

    # Evaluate model on training and validation set every `evaluate_at`
    # updates. This will also check for a new best model for early stopping.
    # Recommended: int(5e3). Default: int(1e2).
    evaluate_at = cfg.training.evaluate_at

    # Size of 1D-CNN kernels (=how many sequence characters a CNN kernel
    # (spans). Default: 9.
    # kernel_size = cfg.model.kernel_size

    # Number of kernels in the 1D-CNN. This is an important hyper-parameter.
    # Default: 32.
    # n_kernels = cfg.model.n_kernels

    # Number of instances to reduce genomes to during training via
    # random dropout. This should be less than the number of instances per
    # genome. Only applied during training, not for evaluation.
    # Default: int(1e4).
    sample_n_sequences = cfg.data_splitting.sample_n_sequences

    # Learning rate of DeepRC using Adam optimizer. Default: 1e-4.
    learning_rate = cfg.training.learning_rate

    # Batch size
    # Default: 4
    batch_size = cfg.training.batch_size

    # Dataloader worker processes
    # Default: 4
    n_worker_processes = cfg.training.n_worker_processes

    # Request for permition to create hdf5
    request_for_permition_to_create_hdf5 = \
        cfg.request_for_permition_to_create_hdf5

    # Device to use for NN computations, as passed to `torch.device()`.
    # Default: 'cuda:0'.
    device = cfg.device

    # Random seed to use for PyTorch and NumPy.
    # Results will still be non-deterministic due to multiprocessing, but
    # weight initialization will be the same.
    # Default: 0.
    rnd_seed = cfg.rnd_seed

    # Create ORFs from JSONs in the `jsons` directory. Default: False.
    create_orfs = cfg.create_orfs

    # General
    results_directory = cfg.results_directory
    metadata_file = path.abspath(cfg.database.metadata_file)
    genomesdata_path = path.abspath(cfg.database.genomesdata_path)
    early_stopping_target_id = cfg.task.target.column_name

    # Data splitting
    stratify = cfg.data_splitting.stratify
    metadata_file_id_column = cfg.data_splitting.metadata_file_id_column
    sequence_column = cfg.data_splitting.sequence_column
    split_column_name = cfg.data_splitting.split_column_name
    sequence_counts_column = cfg.data_splitting.sequence_counts_column

    # Check if database is present
    base_dir = path.abspath(cfg.database.base_dir)
    is_present = check_database_presence("train", base_dir)
    if not is_present:
        raise FileNotFoundError("Database folder/file not found!")

    if create_orfs:
        create_orf_table()

    # Set computation device
    device = torch.device(device)
    # Set random seed (Weight initialization will be the same)
    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    # Create Task definitions

    # Assume we want to train on 1 main task and 5 auxiliary tasks. We will set
    # the task-weight of the main task to 1 and of the auxiliary tasks to
    # 0.1/5. The tasks-weight is used to compute the training loss as weighted
    # sum of the individual tasks losses.
    # aux_task_weight = 0.1 / 5
    # Below we define how the tasks should be extracted from the metadata file.
    # We can choose between combinations of binary, regression, and multiclass
    # tasks. The column names have to be found in the metadata file.
    task_definition = create_task_definition(
        cfg.task).to(device=cfg.device)

    # Get dataset

    # Get data loaders for training set and training-, validation-,
    # and test-set in evaluation mode (=no random subsampling)
    trainingset, trainingset_eval, \
        validationset_eval, testset_eval = make_dataloaders_stratified(
            task_definition=task_definition,
            metadata_file=metadata_file,
            repertoiresdata_path=genomesdata_path,
            split_column_name=split_column_name,
            n_splits=5,
            stratify=stratify,
            rnd_seed=rnd_seed,
            n_worker_processes=n_worker_processes,
            batch_size=batch_size,
            ask_for_input=request_for_permition_to_create_hdf5,
            metadata_file_id_column=metadata_file_id_column,
            sequence_column=sequence_column,
            sequence_counts_column=sequence_counts_column,
            sample_n_sequences=sample_n_sequences,
            # Alternative: deeprc.dataset_readers.log_sequence_count_scaling
            sequence_counts_scaling_fn=no_sequence_count_scaling
        )

    # Create DeepRC Network
    model = build_model(cfg)

    # Train DeepRC model
    train(model, task_definition=task_definition,
          trainingset_dataloader=trainingset,
          trainingset_eval_dataloader=trainingset_eval,
          learning_rate=learning_rate,
          # Get model that performs best for this task
          early_stopping_target_id=early_stopping_target_id,
          validationset_eval_dataloader=validationset_eval,
          n_updates=n_updates, evaluate_at=evaluate_at,
          # Here our results and trained models will be stored
          device=device, results_directory=results_directory
          )  # type: ignore

    scores = evaluate(model=model, dataloader=testset_eval,
                      task_definition=task_definition, device=device)
    print(f"Test scores:\n{scores}")

    clear_gpu_memory()


if __name__ == "__main__":
    main()
