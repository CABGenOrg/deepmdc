import pandas as pd
from functools import partial
from os import path, listdir, mkdir
from concurrent.futures import ProcessPoolExecutor
from src.utils.handle_files import write_table, read_tsv
from src.utils.handle_processing import process_json, process_tsv


def create_orf_table(metadata_file="database/metadata.tsv",
                     complete_table_dir="database/complete_tables",
                     jsons_dir="jsons",
                     orf_tables_dir="database/orfs",
                     filter_orfs=False):
    """
    Creates ORF tables by processing genomic metadata and extracting ORF
    information from JSON files.

    This function reads sample IDs from a metadata file, retrieves
    corresponding ORF data from JSON files, compiles the information into
    structured tables, and applies optional filtering to the ORF sequences.

    Args:
        metadata_file (str): Path to TSV file containing sample IDs and
        associated metadata. The file should contain the identifiers used to
        match JSON files.
        complete_table_dir (str): Directory path where the complete compiled
        ORF table will be saved.
        json_dir (str): Directory path containing JSON files with ORF
        information. JSON files are expected to be named matching the IDs from
        metadata_file.
        orf_tables_dir (str): Directory path where individual ORF tables will
        be saved.
        filter_orfs (bool, optional): If True, applies filtering to remove ORFs
        annotated as "hypothetical protein" and those with amino acid lengths
        outside the 40-2000 aa range. Defaults to False.
    """
    metadata_file = path.abspath(metadata_file)
    metadata = pd.read_csv(metadata_file, sep='\t')
    ids_to_use = {id.split(".")[0] for id in metadata['ID']}

    complete_table_dir = path.abspath(complete_table_dir)
    if not path.exists(complete_table_dir):
        mkdir(complete_table_dir)

    base_dir = path.abspath(jsons_dir)
    json_files = [path.join(base_dir, json) for json in listdir(base_dir)
                  if json.endswith("json.xz") and
                  json.split(".json")[0] in ids_to_use]

    if len(json_files) == 0:
        return

    worker = partial(json_to_table, complete_table_dir=complete_table_dir,
                     orf_tables_dir=orf_tables_dir, filter_orfs=filter_orfs)
    with ProcessPoolExecutor() as executor:
        executor.map(worker, json_files)

    return


def json_to_table(json_file: str, complete_table_dir: str,
                  orf_tables_dir: str, filter_orfs: bool = False):
    """
    Converts ORF data from a JSON file into TSV table format and saves the
    results.

    This function processes a single JSON file containing ORF information,
    extracts relevant data, converts it to tabular format, and saves both
    individual and compiled ORF tables. Optional filtering can be applied to
    exclude certain ORFs based on annotation and length criteria.

    Args:
        json_file (str): Path to the JSON file containing ORF information to be
        processed.
        complete_table_dir (str): Directory path where the compiled ORF table
        (aggregating data from all processed files) will be saved as a TSV
        file.
        orf_tables_dir (str): Directory path where individual ORF tables (one
        per JSON file) will be saved as TSV files.
        filter_orfs (bool, optional): If True, filters out ORFs annotated as
        "hypothetical protein" and those with amino acid lengths outside the
        40-2000 range. Defaults to False.
    """
    rows = process_json(json_file, filter_orfs)

    table_name = f"{json_file.split('/')[-1].split('.')[0]}.tsv"
    complete_table = path.join(complete_table_dir, table_name)

    write_table(complete_table, rows)

    orf_table = path.abspath(path.join(orf_tables_dir, table_name))
    df = read_tsv(complete_table)

    orf_df = process_tsv(df)
    write_table(orf_table, orf_df)
    return
