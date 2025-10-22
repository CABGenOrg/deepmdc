import hydra
import torch
from shutil import rmtree
from omegaconf import DictConfig
from deeprc.training import evaluate
from os import path, environ, listdir
from cabgen_hopfield_main import create_task_definition
from src.utils.handle_files import get_most_recent_folder
from src.utils.handle_machine_learning import build_model
from widis_lstm_tools.utils.collection import SaverLoader
from src.utils.handle_files import check_database_presence
from src.utils.handle_processing import make_dataloader


environ["PYTORCH_CUDA_ALLOC_CONF"] = ("garbage_collection_threshold:0.8,"
                                      "max_split_size_mb:128,"
                                      "expandable_segments:True")
environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="config", config_name="config")
def test_model(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Test config
    model_path = cfg.test.model_path
    metadata_file = path.abspath(cfg.test.metadata_file)
    genomesdata_path = path.abspath(cfg.test.genomesdata_path)
    request_for_permition_to_create_hdf5 = \
        cfg.request_for_permition_to_create_hdf5

    model = build_model(cfg)
    save_dir = "tmp"
    state = dict(model=model)
    saver_loader = SaverLoader(save_dict=state, device=cfg.device,
                               save_dir=save_dir)

    # Load model state
    if "/" not in model_path or not path.exists(model_path):
        model_folder = path.join(get_most_recent_folder(model_path),
                                 "checkpoint")
        model_path = [path.join(model_folder, file)
                      for file in listdir(model_folder)][0]

    print(f"Loading model from {model_path}...")
    saver_loader.load_from_file(loadname=model_path, verbose=True)
    print("Model loaded successfully.")

    # Set model to evaluation mode
    model.eval()

    print("Preparing the test data...")
    is_present = check_database_presence("test", ".")
    if not is_present:
        raise FileNotFoundError("Database test folder/file not found!")

    task_definition = create_task_definition(
        cfg.task).to(device=cfg.device)

    testset_eval = make_dataloader(
        task_definition=task_definition, metadata_file=metadata_file,
        genomesdata_path=genomesdata_path,
        ask_for_input=request_for_permition_to_create_hdf5)

    print("Evaluating the model on test data...")
    scores = evaluate(model=model, dataloader=testset_eval,
                      task_definition=task_definition, device=device)

    print("\nTest set results:")
    print(scores)
    rmtree(save_dir)


if __name__ == "__main__":
    test_model()
