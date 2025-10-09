import lzma
import pandas as pd
from json import load
from shutil import rmtree
from subprocess import run
from typing import List, Union
from os import path, listdir, remove


def read_json(json_path: str):
    """
    Reads a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Any: Data from the JSON file.
    """
    try:
        with open(json_path) as inp:
            json_data = load(inp)
        return json_data
    except Exception as error:
        raise Exception(f"Can't read json file...\n{error}")


def write_table(table_name: str, data: Union[List[List[Union[str, int]]],
                                             pd.DataFrame]) -> None:
    """
    Writes ORF (Open Reading Frame) information to a table file.

    This function creates a table containing information about identified ORFs,
    such as their coordinates, length, sequence, and other relevant metadata.

    Args:
        table_name (str): The name or path of the output table file.
        data (List[List[Union[str, int]]] | pd.DataFrame): The ORF data to be
        written to the table. Can be either a list of lists (where each
        inner list represents a row) or a pandas DataFrame containing
        ORF information.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data, columns=["locus", "type", "product",
                                         "sequence", "count"])
    df.to_csv(table_name, sep="\t", index=False)


def read_tsv(table_file: str) -> pd.DataFrame:
    """
    Reads data from a TSV (Tab-Separated Values) file into a pandas DataFrame.

    This function handles the parsing of a TSV file, expecting the first row to
    contain column headers.

    Args:
        table_file (str): The path or filename of the TSV file to be read.

    Returns:
        pd.DataFrame: A pandas DataFrame object containing all the data from
        the TSV file.
    """
    df = pd.read_csv(table_file, sep="\t")
    return df


def get_most_recent_folder(folder_path: str) -> str:
    """
    Gets the folder path of the most recent folder inside a folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        str: Path of the most recent folder.
    """
    folders = [path.join(folder_path, folder) for folder in
               listdir(folder_path) if path.isdir(
                   path.join(folder_path, folder))]
    if not folders:
        return ""

    most_recent_file_path = max(folders, key=path.getctime)
    return most_recent_file_path


def balance_samples(old_data: dict, new_data: dict, total: int):
    """
    Balances a metadata table by adding new samples while maintaining class
    balance and ensuring the total number of samples per label does not
    exceed the specified limit.

    Args:
        old_data (dict): Old metadata data in a dictionary.
        new_data (dict): New metadata data in a dictionary.
        total (dict): Total of samples per label.
    """
    old_total_R = old_data.get("count", {}).get("R", 0)
    old_total_S = old_data.get("count", {}).get("S", 0) + \
        old_data.get("count", {}).get("I", 0)
    old_total = old_total_R + old_total_S

    total_to_add = total - old_total
    if total_to_add <= 0:
        raise ValueError("Total cannot be less or equal to zero.")

    target_R = total
    target_S = total

    # Defining the total of samples to add per label
    total_R_to_add = target_R - old_total_R
    total_S_to_add = target_S - old_total_S

    new_total_R = new_data.get("count", {}).get("R", 0)
    new_total_I = new_data.get("count", {}).get("I", 0)
    new_total_S = new_data.get("count", {}).get("S", 0)

    if new_total_R < total_R_to_add:
        raise ValueError("There is no sufficient R samples.")
    elif new_total_S + new_total_I < total_S_to_add:
        raise ValueError("There is no sufficient S samples.")

    if old_data == {}:
        old_data.update({"count": {"R": 0, "S": 0, "I": 0}})

    # Making sure that some I samples will be added (limited to 20% of R total)
    if new_total_I != 0:
        new_data_only_I = {k: v for k, v in new_data.items()
                           if k != "count" and v["real_label"] == "I"}
        I_limit = int(total_S_to_add * 0.2)
        I_count = 0

        for k, v in new_data_only_I.items():
            if k not in old_data and k != "count":
                if total_S_to_add > 0:
                    old_data.update({k: v})
                    old_data["count"]["I"] += 1
                    total_S_to_add -= 1
                    I_count += 1

            if I_limit <= I_count:
                break

    for k, v in new_data.items():
        if k not in old_data and k != "count":
            if v["real_label"] == "R" and total_R_to_add > 0:
                old_data.update({k: v})
                old_data["count"]["R"] += 1

                total_R_to_add -= 1
            elif v["real_label"] == "S" and total_S_to_add > 0:
                old_data.update({k: v})
                old_data["count"]["S"] += 1
                total_S_to_add -= 1

    total = old_data['count']['S'] + \
        old_data['count']['R'] + old_data['count']['I']
    print(f"S: {old_data['count']['S']} | I: {old_data['count']['I']}"
          f" | R: {old_data['count']['R']} | Total: {total}")

    return old_data


def update_metadata(old_metadata: str, new_metadata: str,
                    total_of_samples_per_label: int,
                    atb_column: str, final_metadata_path: str) -> None:
    """
    Updates the metadata table with new samples. Adds the same number of R and
    S samples respecting the desire total of samples per label.

    Args:
        old_metadata (str): Old metadata table path.
        new_metadata (str): New metadata table path.
        total_of_samples_per_label (int): Desired total of samples per label.
        atb_column (str): Antibiotic column name.
        final_metadata_path (str): Final metadata path to write the TSV table.
    """
    if new_metadata == "":
        raise ValueError("New metadata table does not exist.")

    if old_metadata == "":
        old_data = {}
    else:
        old_data = tsv_to_dict(old_metadata)

    new_data = tsv_to_dict(new_metadata)
    data = balance_samples(old_data, new_data, total_of_samples_per_label)
    dict_to_tsv(data, atb_column, final_metadata_path)


def zip_folder(zip_filename: str, folder: str,
               delete_zipped_files=False) -> str:
    """
    Creates a zip file containing the specified list of files.

    Args:
        zip_filename (str): The name of the zip file to be created.
        folder (str): Folder to be zipped.
        delete_zipped_files (bool): Whether to delete the files after creating
        the zip file.

    Returns:
        str: The path of the created zip file.
    """
    command_line = f"tar -cf - {folder} | xz -9e -c > {zip_filename}"
    run(command_line, shell=True)

    if delete_zipped_files:
        rmtree(folder)

    return zip_filename


def unzip(file: str, delete_zip_file=False):
    """
    Unzips a zip file.

    Args:
        file (str): The name of the zip file to be unzipped.
        delete_zip_file (bool): Whether to delete the zip file after unzip.
    """
    command_line = f"tar -xf {file}"
    run(command_line, shell=True)

    if delete_zip_file:
        remove(file)
    return


def check_database_presence(database_type: str, base_dir: str) -> bool:
    """
    Verifies the presence of the directory containing training or testing data.

    This function checks whether the required data directory exists for the
    specified database type, which is essential for ensuring data availability
    before initiating training or testing procedures.

    Args:
        database_type (str): Type of database to check. Expected values are
        "train" for training data (/database) or "test" for testing data
        (/test).

    Returns:
            bool: True if the corresponding data directory exists and is
            accessible, False otherwise.
    """
    if database_type == "train":
        folder_to_search = "database"
        file_extension = ".tar.xz"
    elif database_type == "test":
        folder_to_search = "test"
        file_extension = "_test.tar.xz"
    else:
        raise ValueError("Invalid database type. Choose 'train' or 'test'.")

    files = listdir(base_dir)
    database_file = [file for file in files if file.endswith(file_extension)]

    if folder_to_search in files:
        return True
    elif len(database_file) != 0:
        unzip(database_file[0], True)
        return True
    else:
        return False


def read_compressed_json(json_path: str):
    """
    Reads a compressed JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        Any: Data from the JSON file.
    """
    if not path.exists(json_path):
        return {}

    with lzma.open(json_path, "rt", encoding="utf-8") as inp:
        json_data = load(inp)

    return json_data


def tsv_to_dict(tsv_path: str) -> dict:
    """
    Converts raw metadata TSV to a dictionary to be processed.

    Args:
        tsv_path (str): TSV path.
    Returns:
        dict: Metadata dict.
    """
    dic = {"count": {}}
    with open(tsv_path) as inp:
        for line in inp.read().splitlines()[1:]:
            id = ""
            atb = ""
            real_atb = ""

            rows = [cel for cel in line.split("\t") if cel != ""]
            if len(rows) == 2:
                id = rows[0].strip()
                atb = rows[1].strip()
                real_atb = atb
            else:
                id = rows[0].strip()
                atb = rows[1].strip()
                real_atb = rows[2].strip()

            dic.update({id: {"label": atb, "real_label": real_atb}})

            if real_atb in dic["count"]:
                dic["count"][real_atb] += 1
            else:
                dic["count"][real_atb] = 1

    return dic


def dict_to_tsv(dic: dict, atb_column: str, tsv_path: str):
    """
    Creates a TSV table from a metadata dictionary.

    Args:
        dic (dict): Dictionary metadata.
        atb_column (str): Antibiotic column name.
        tsv_path (str): TSV path to write.
    Returns:
        dict: Metadata dict.
    """
    columns = ["ID", atb_column]

    data = sorted([[f'{k}.tsv', v.get("label", "")] for k, v in dic.items()
                   if k != "count"], key=lambda x: x[1])

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(tsv_path, sep="\t", index=False)
