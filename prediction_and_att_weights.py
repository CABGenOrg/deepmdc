import hydra
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import rmtree
from os import path, listdir
from omegaconf import DictConfig
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_files import get_most_recent_folder, read_compressed_json
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_machine_learning import build_model
from src.utils.handle_processing import make_dataloader


def enrich_predictions_with_json(
        predictions_df: pd.DataFrame, json_data: dict) -> pd.DataFrame:
    """
    Enriches the prediction DataFrame with product and db_xrefs information
    from a JSON data source.
    """
    print("Enriching prediction data with JSON info...")
    enriched_rows = []

    for row in tqdm(predictions_df.itertuples(index=False),
                    total=len(predictions_df), desc="Processing rows"):
        temp_dic = {
            "id": row.ID,
            "inference": row.inference,
            "prediction": row.prediction,
            "label": row.label,
            "match": row.match,
        }

        # Adding sequence and product information
        for i in range(1, 6):
            seq_key = f"seq_{i}"
            weight_key = f"weight_{i}"

            # Ensuring the columns exist in the original DataFrame
            if hasattr(row, seq_key) and hasattr(row, weight_key):
                sequence = getattr(row, seq_key)
                temp_dic[f"seq{i}"] = sequence
                temp_dic[f"weight{i}"] = getattr(row, weight_key)

                # Retrieving the JSON data safely
                json_entry = json_data.get(sequence, {})
                temp_dic[f"product{i}"] = json_entry.get("product", "")
                temp_dic[f"db_xrefs{i}"] = json_entry.get("ids", "")

        enriched_rows.append(temp_dic)

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.sort_values(by=["label", "match"],
                            ascending=False, inplace=True)

    print("Data enrichment complete.")
    return enriched_df


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """
    Generates predictions, enrich them with additional data, and save the
    final processed table.
    """
    # Device configuration
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Input paths
    model_path = cfg.test.model_path
    metadata_file = path.abspath(cfg.test.metadata_file)
    genomesdata_path = path.abspath(cfg.test.genomesdata_path)
    json_file = path.abspath(cfg.biological_json_file)

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

    model.eval()

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

    with torch.no_grad():
        for scoring_data in tqdm(testset_eval, total=len(testset_eval),
                                 desc="Evaluating model"):
            targets, inputs, sequence_lengths, \
                counts_per_sequence, sample_ids = scoring_data

            aa_sequences_list = [
                inds_to_aa_ignore_negative(aa) for aa in inputs[0]]

            label, inputs, sequence_lengths, n_sequences, \
                used_sequence_inds = \
                model.reduce_and_stack_minibatch(
                    targets, inputs, sequence_lengths, counts_per_sequence)

            raw_outputs = model(inputs_flat=inputs,
                                sequence_lengths_flat=sequence_lengths,
                                n_sequences_per_bag=n_sequences)
            probabilities = torch.sigmoid(raw_outputs).cpu().numpy().flatten()
            used_sequence_inds_list = used_sequence_inds[0].tolist()

            embeddings = model.sequence_embedding(
                inputs, sequence_lengths=sequence_lengths).to(
                    dtype=torch.float32
            )
            attention_logits = model.attention_nn(embeddings)
            attention_weights = torch.softmax(attention_logits, dim=0).t()
            attention_weights_list = attention_weights.tolist()[0]

            prediction = "R" if probabilities[0] >= 0.5 else "S"
            classification = "R" if label.cpu().numpy().flatten() else "S"
            all_predictions.append(
                {
                    "ID": sample_ids[0],
                    "inference": probabilities[0],
                    "prediction": prediction,
                    "label": classification,
                    "match": "Y" if prediction == classification else "N",
                    "seq_1": aa_sequences_list[used_sequence_inds_list[0]],
                    "weight_1": attention_weights_list[0],
                    "seq_2": aa_sequences_list[used_sequence_inds_list[1]],
                    "weight_2": attention_weights_list[1],
                    "seq_3": aa_sequences_list[used_sequence_inds_list[2]],
                    "weight_3": attention_weights_list[2],
                    "seq_4": aa_sequences_list[used_sequence_inds_list[3]],
                    "weight_4": attention_weights_list[3],
                    "seq_5": aa_sequences_list[used_sequence_inds_list[4]],
                    "weight_5": attention_weights_list[4],
                }
            )

    predictions_df = pd.DataFrame(all_predictions)
    print("Initial prediction table generated in memory.")

    print(f"Loading ORF data from {json_file}...")
    json_data = read_compressed_json(json_file)

    final_df = enrich_predictions_with_json(predictions_df, json_data)

    output_file = (f"bio_{antibiotic_name}_{model_path.split('/')[1]}"
                   "_enriched_predictions.tsv")
    final_df.to_csv(output_file, index=False, sep="\t")

    print(f"Final enriched predictions saved to {output_file}")
    rmtree(output_dir)


def inds_to_aa_ignore_negative(inds: torch.Tensor) -> str:
    """
    Converts array of AA indices to character string, ignoring '-1'-padding.
    """
    aas = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
    inds_array = inds.detach().cpu().numpy()
    valid_inds = inds_array[inds_array >= 0]
    aa_map = np.array(aas)
    valid_aas = aa_map[valid_inds]
    return ''.join(valid_aas.tolist())


if __name__ == "__main__":
    main()
