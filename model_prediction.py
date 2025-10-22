import hydra
import torch
import pandas as pd
from tqdm import tqdm
from shutil import rmtree
from os import path, listdir
from omegaconf import DictConfig
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_files import get_most_recent_folder
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_machine_learning import build_model
from src.utils.handle_files import check_database_presence
from src.utils.handle_processing import make_dataloader, \
    insert_datetime_into_filename


@hydra.main(version_base=None, config_path="config", config_name="config")
def predict_samples(cfg: DictConfig):
    """Generate predictions for test samples and save them as a table."""

    # Device configuration
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Input paths
    model_path = cfg.test.model_path
    metadata_file = path.abspath(cfg.test.metadata_file)
    genomesdata_path = path.abspath(cfg.test.genomesdata_path)

    # Get the antibiotic name to use as the column name
    antibiotic_name = cfg.task.target.column_name

    model = build_model(cfg)
    # Load saved model weights
    print("Loading model...")
    output_dir = "tmp"
    state = dict(model=model)
    saver_loader = SaverLoader(save_dict=state, device=cfg.device,
                               save_dir=output_dir)

    if "/" not in model_path or not path.exists(model_path):
        model_folder = path.join(
            get_most_recent_folder(model_path), "checkpoint")
        model_path = [path.join(model_folder, file)
                      for file in listdir(model_folder)][0]

    saver_loader.load_from_file(loadname=model_path, verbose=True)
    print(f"Model loaded successfully from {model_path}.")

    # Set the model to evaluation mode
    model.eval()

    # Prepare test data
    print("Preparing test data...")
    is_present = check_database_presence("test", ".")
    if not is_present:
        raise FileNotFoundError("Database test folder/file not found!")
    task_definition = create_task_definition(cfg.task).to(device=device)
    request_for_permition_to_create_hdf5 = \
        cfg.request_for_permition_to_create_hdf5

    testset_eval = make_dataloader(
        task_definition=task_definition,
        metadata_file=metadata_file,
        genomesdata_path=genomesdata_path,
        ask_for_input=request_for_permition_to_create_hdf5
    )

    print("Generating predictions...")
    all_predictions = []

    # Iterate over test data
    with torch.no_grad():
        for scoring_data in tqdm(testset_eval, total=len(testset_eval),
                                 desc="Evaluating model"):

            # Extract batch data
            targets, inputs, sequence_lengths, \
                counts_per_sequence, sample_ids = scoring_data

            # Apply attention-based sequence reduction and create a minibatch
            _, inputs, sequence_lengths, n_sequences, _ = \
                model.reduce_and_stack_minibatch(
                    targets, inputs, sequence_lengths, counts_per_sequence)

            # Forward pass to obtain predictions
            raw_outputs = model(inputs_flat=inputs,
                                sequence_lengths_flat=sequence_lengths,
                                n_sequences_per_bag=n_sequences)
            probabilities = torch.sigmoid(raw_outputs).cpu().numpy().flatten()

            # Store the results
            for sample_id, prob in zip(sample_ids, probabilities):
                all_predictions.append(
                    {"ID": sample_id, antibiotic_name: prob})

    # Create DataFrame and save as TSV
    df = pd.DataFrame(all_predictions)
    output_file = insert_datetime_into_filename(
        f"{path.basename(model_path).split('.')[0]+'_'+antibiotic_name}"
        "_predict.tsv")
    df.to_csv((output_file), index=False, sep="\t")

    print(f"Predictions saved to {output_file}")
    rmtree(output_dir)


if __name__ == "__main__":
    predict_samples()
