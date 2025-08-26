# Deep Multiple Drug Classification

## Requirements

To run the script correctly, Python 3.10.12 is required. Additionally, having an NVIDIA GPU is recommended, as it can significantly speed up the training process.

### Setup

The environment can be set up using either Python or Conda. In both cases, it is crucial to activate the environment before running any commands.

#### Python

To set up the environment and install all the required dependencies, run the following command:

```bash
make setup
```

Next, activate the environment:

```bash
source ./.venv/bin/activate
```

#### Conda

First, install Miniconda in your environment by following the [documentation](https://www.anaconda.com/docs/getting-started/miniconda/install). Once the installation is complete, create a new environment with Python 3.10.12 and activate it.

```bash
conda create -n deepmdc python=3.10.12
conda activate deepmdc
```

Finally, install the Python libraries:

```bash
pip install -r requirements.txt
```

## Hydra Settings

Hydra configurations make it easy to manage training and testing parameters. Below is a description of the main configuration files and their parameters.

* **config.yaml**:
Main configuration file, includes references to other config files and global parameters.

  * **device**: Specifies the device to run the model on. Example: `"cuda:0"` to use GPU.
  * **rnd_seed**: Sets the random seed for reproducibility.
  * **results_directory**: Directory where training results will be saved. Default: `"results"`.
  * **max_seq_len**: Maximum length of input sequences.
  * **database**: Contains paths to input files:
    * **metadata_file**: TSV file with sample IDs and resistance profiles. Default: `"database/metadata.tsv"`.
    * **repertoiresdata_path**: Directory containing ORF files. Default: `"database/orfs.hdf5"`.

* **data_splitting.yaml**:
Configuration for data splitting.

  * **stratify**: If true, performs stratified splitting to balance classes.
  * **metadata_file_id_column**: Column name in metadata containing sample IDs.
  * **sequence_column**: Column name containing ORF sequences in input files.
  * **sequence_counts_column**: Column indicating how many times each ORF appears.
  * **sample_n_sequences**: Number of sequences to sample per genome during training. Use 0 to load all sequences.

* **model.yaml**:
Model architecture configuration.

  * **sequence_embedding**:
    * **type**: Embedding layer type, e.g., CNN or LSTM.
    * **n_layers**: Number of layers in sequence embedding.
    * **kernel_size**: CNN kernel size (number of amino acids per filter).
    * **n_units**: Number of neurons in embedding layer (number of kernels or LSTM blocks).

  * **attention**:
    * **n_layers**: Number of layers in attention network.
    * **n_units**: Number of neurons in attention network.

  * **output**:
    * **n_layers**: Number of layers in output network.
    * **n_units**: Number of neurons in output network.

* **task.yaml**:
Task configuration.

  * **target**: Defines classification tasks:

    * **type**: Task type, e.g., binary.
    * **column_name**: Metadata column with task labels.
    * **positive_class**: For binary tasks, defines the positive class.
    * **pos_weight**: Weight applied to the positive class to handle imbalance.
    * **task_weight**: Weight of this task in the total loss function (if multiple tasks).

* **training.yaml**:
Training configuration.

  * **n_updates**: Number of update steps during training.
  * **evaluate_at**: Frequency (in steps) to evaluate the model on validation and training sets.
  * **learning_rate**: Learning rate for the Adam optimizer.

* **test.yaml**:
Model evaluation configuration.

  * **model_path**: Path to the trained model to be evaluated. Default: `"results"`.
  * **metadata_file**: Path to test metadata file.
  * **orfs_path**: Path to ORF files for the test set.

## Input Format

Two files are required for both training and testing the models.

**metadata.tsv:** This file contains the filenames and their classification under a specific label.

| ID          | MEM |
| ----------- | --- |
| sample1.tsv | R   |
| sample2.tsv | S   |

**sample1.tsv:** Each of these files represents the genomic repertoire of a different bacterial sample. These files contain a column with the ORF sequence and a second column indicating how many times it appears in the genome.

| orf                              | templates |
| -------------------------------- | --------- |
| MKKTRYTEEQIAFALKQAE              | 1         |
| MTGICVGEVCRKMGISEAIFIIRREIRCSGRN | 2         |

## Datasets Download

The datasets used to train the models described in the article can be found in this [repository](https://github.com/CABGenOrg/deepmdc_data).
Before running the training or testing, you need to:

1. Download the datasets files.
2. Unzip the contents.
3. Place them in the **root folder** of the project.

## Usage

### Training

To start training with the default parameters, simply run:

```bash
make run
```

You can also customize the training configuration by passing parameters directly. For example:

* **Adjust the learning rate and the number of sequence embedding units:**

```bash
python3 cabgen_hopfield_main.py training.learning_rate=0.001 model.sequence_embedding.n_units=64
```

* **Combine multiple parameters to train several models at once**:

```bash
python3 cabgen_hopfield_main.py -m model.sequence_embedding.kernel_size=30,45 training.learning_rate=0.00005 training.n_updates=10000 training.evaluate_at=100,200,500 data_splitting.sample_n_sequences=0 +hydra.job.logging=debug hydra.verbose=true hydra/launcher=basic
```

### Testing

The script `test_model.py` evaluates a trained model on new data.
By default, it uses the settings defined in `config/test.yaml` and evaluates the most recent model stored in `/results`.
To test a different model, adjust the parameter `test.model_path`.

* **Evaluate the most recent model in** `/results`:

```bash
python3 test_model.py
```

* **Evaluate a specific model (e.g., with kernel size 48):**

```bash
python3 test_model.py test.model_path="results/model_2050125/checkpoint/model.zip" model.kernel_size=48
```

* **Change the test metadata file:**

```bash
python3 test_model.py test.metadata_file="new_metadata.tsv"
```

### 5 Fold Cross Validation

The script fold_cross_validation.py performs 5-fold cross-validation to assess the stability of model metrics.
It splits the dataset into 5 subsets and trains 5 models, each time varying the validation/test subset.
You can also override any parameter directly from the command line, depending on the script’s available options.

* **Run cross-validation with the default settings:**

```bash
python3 fold_cross_validation.py
```

* **Run cross-validation with a kernel size of 48 and 10,000 updates:**

```bash
python3 fold_cross_validation.py model.sequence_embedding.kernel_size=48 training.n_updates=10000
```

## Model Overview

1. **Data splitting:**

The function `make_dataloaders_stratified` splits the dataset into training, validation, and test sets.
When the `stratify` option is enabled, it ensures balanced class distributions across splits.
For example, if the class proportion is 50/50, each split will preserve that ratio (or as close as possible).

2. **Model architecture:**

The **DeepRC** model integrates three main components:

* **Sequence Embedding:** Captures patterns in ORF sequences (using CNN or LSTM).
* **Attention Network:** A modern Hopfield network that highlights the most relevant ORF regions.
* **Output Network:** Classifies the samples into the target classes.

3. **Training:**

During training, the model adjusts its weights to minimize the loss function.

4. **Evaluation:**

The model is evaluated using the following metrics:

* **ROC AUC**
* **Balanced Accuracy (BACC)**
* **F1-score**
* **Matthews Correlation Coefficient (MCC)**
* **Loss**
