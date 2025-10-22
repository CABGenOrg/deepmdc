import hydra
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import rmtree
from os import path, listdir
from omegaconf import DictConfig
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_files import get_most_recent_folder
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_machine_learning import build_model
from src.utils.handle_processing import make_dataloader


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
    antibiotic_name = cfg.task['target']['column_name']

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

            # Get aa sequences list
            aa_sequences_list = [inds_to_aa_ignore_negative(aa)
                                 for aa in inputs[0]]

            # Apply attention-based sequence reduction and create a minibatch
            _, inputs, sequence_lengths, n_sequences, used_sequence_inds = \
                model.reduce_and_stack_minibatch(
                    targets, inputs, sequence_lengths, counts_per_sequence)

            # Transform sequence indices in list
            used_sequence_inds_list = used_sequence_inds[0].tolist()

            # Apply CNN Attention Network and softmax to obtain weights
            embeddings = model.sequence_embedding(
                inputs, sequence_lengths=sequence_lengths).to(
                    dtype=torch.float32)

            attention_logits = model.attention_nn(embeddings)
            attention_weights = torch.softmax(attention_logits, dim=0).t()
            attention_weights_list = attention_weights.tolist()[0]

            # Store the results
            all_predictions.extend([
                {
                    "ID": sample_ids[0],
                    "att_weight": weight,
                    "sequence": aa_sequences_list[ind]
                }
                for ind, weight in zip(used_sequence_inds_list,
                                       attention_weights_list)
            ])

            all_predictions.extend([
                {
                    "ID": sample_ids[0],
                    "att_weight": 0.0,
                    "sequence": seq
                }
                for ind, seq in enumerate(aa_sequences_list)
                if ind not in used_sequence_inds_list
            ])

    # Create DataFrame and save as TSV
    df = pd.DataFrame(all_predictions)
    output_file = (f"{antibiotic_name}_{model_path.split('/')[1]}"
                   "_attention_weights.tsv")
    df.to_csv((output_file), index=False, sep="\t")

    print(f"Predictions saved to {output_file}")
    rmtree(output_dir)


def inds_to_aa_ignore_negative(inds: torch.Tensor) -> str:
    """
    Convert array of AA indices to character string, ignoring '-1'-padding
    to equal sequence length

    Args:
        inds: Tensor containing amino acid indices (-1 for padding)

    Returns:
        String of amino acid characters
    """
    aas = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')

    # Convert to numpy array and filter out padding (-1)
    inds_array = inds.detach().cpu().numpy()
    valid_inds = inds_array[inds_array >= 0]

    # Create mapping and convert to string
    aa_map = np.array(aas)
    valid_aas = aa_map[valid_inds]

    return ''.join(valid_aas.tolist())


if __name__ == "__main__":
    predict_samples()
