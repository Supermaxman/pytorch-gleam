import argparse
import csv
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm

# Increase field limit to avoid csv.Error: field larger than field limit (131072)
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
csv.field_size_limit(sys.maxsize)


def read_file(file_path: str, codes: Set[str]) -> Dict[str, List[str]]:
    current = mp.current_process()
    pos = current._identity[0] - 1

    # each .txt file is a csv with | as delimiter
    # the columns are provided for the first row as:
    # PTID|ENCID|DIAG_DATE|DIAG_TIME|DIAGNOSIS_CD|DIAGNOSIS_CD_TYPE|DIAGNOSIS_STATUS|POA|ADMITTING_DIAGNOSIS|DISCHARGE_DIAGNOSIS|PRIMARY_DIAGNOSIS|PROBLEM_LIST|SOURCEID
    # 72402|ICD9|Diagnosis of|0|0|0|1|N|S0034
    date_patients: Dict[str, List[str]] = defaultdict(list)
    file_name = os.path.basename(file_path).split(".")[0]
    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        # estimate from the first file that there are 41,461,047 rows
        for row in tqdm(reader, total=236_075_047, desc=file_name, position=pos, miniters=236_075_047 / 1_000):
            # dates are in the format MM-DD-YYYY (e.g. 07-14-2021)
            if row["DIAGNOSIS_CD"] in codes:
                date = row["DIAG_DATE"]
                date = datetime.strptime(date, "%m-%d-%Y")
                date = date.strftime("%Y-%m-%d")
                date_patients[date].append(row["PTID"])
    return date_patients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/shared/aifiles/disk1/covid19/raw/rawdata/20220120")
    parser.add_argument("--output_dir", type=str, default="/shared/aifiles/disk1/covid19/artifacts/diag_over_time")
    parser.add_argument("--prefix", type=str, default="cov_20220120_diag_")

    args = parser.parse_args()
    diagnosis_codes = {"B9729", "J1282", "U071", "U072", "U07", "Z8616", "Z20822", "Z20828"}
    # get all files in data_dir
    files = sorted(
        [file for file in os.listdir(args.data_dir) if file.endswith(".txt") and file.startswith(args.prefix)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # mkdir for output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # read files in parallel
    date_patients: Dict[str, List[str]] = defaultdict(list)
    with mp.Pool(processes=len(files), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = pool.starmap(read_file, [(os.path.join(args.data_dir, file), diagnosis_codes) for file in files])
        for result in results:
            for date, patients in result.items():
                date_patients[date].extend(patients)

    # write output
    with open(os.path.join(args.output_dir, "diag_patients_over_time.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "PTID"])
        for date, patients in tqdm(date_patients.items(), desc="Writing output"):
            for patient in patients:
                writer.writerow([date, patient])


if __name__ == "__main__":
    main()
