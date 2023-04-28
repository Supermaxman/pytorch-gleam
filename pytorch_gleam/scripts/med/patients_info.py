import argparse
import csv
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict

from tqdm import tqdm

# Increase field limit to avoid csv.Error: field larger than field limit (131072)
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
csv.field_size_limit(sys.maxsize)


def read_file(file_path: str) -> Dict[str, Dict[str, str]]:
    current = mp.current_process()
    pos = current._identity[0] - 1

    # each .txt file is a csv with | as delimiter
    # the columns are provided for the first row as:
    # PTID|BIRTH_YR|GENDER|RACE|ETHNICITY|REGION|DIVISION|DECEASED_INDICATOR|DATE_OF_DEATH|PROVID_PCP|IDN_INDICATOR|FIRST_MONTH_ACTIVE|LAST_MONTH_ACTIVE|NOTES_ELIGIBLE|HAS_NOTES|SOURCEID|SOURCE_DATA_THROUGH
    patients: Dict[str, Dict[str, str]] = {}
    file_name = os.path.basename(file_path).split(".")[0]
    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        # estimate from the first file that there are 41,461,047 rows
        for row in tqdm(reader, total=88_7401, desc=file_name, position=pos, miniters=88_7401 / 1_000):
            # dates are in the format MM-DD-YYYY (e.g. 07-14-2021)
            date_of_death = row["DATE_OF_DEATH"]
            if date_of_death:
                date_of_death = datetime.strptime(date_of_death, "%Y%m")
                date_of_death = date_of_death.strftime("%Y-%m")
            patients[row["PTID"]] = {
                "BIRTH_YR": row["BIRTH_YR"],
                "GENDER": row["GENDER"],
                "RACE": row["RACE"],
                "ETHNICITY": row["ETHNICITY"],
                "REGION": row["REGION"],
                "DIVISION": row["DIVISION"],
                "DECEASED_INDICATOR": row["DECEASED_INDICATOR"],
                "DATE_OF_DEATH": date_of_death,
            }
    return patients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/shared/aifiles/disk1/covid19/raw/rawdata/20220120")
    parser.add_argument("--output_dir", type=str, default="/shared/aifiles/disk1/covid19/artifacts/patients_info")
    parser.add_argument("--prefix", type=str, default="cov_20220120_pt_")

    args = parser.parse_args()
    # get all files in data_dir
    files = sorted(
        [file for file in os.listdir(args.data_dir) if file.endswith(".txt") and file.startswith(args.prefix)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # mkdir for output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # read files in parallel
    patients: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(list))
    with mp.Pool(processes=len(files), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = pool.starmap(read_file, [(os.path.join(args.data_dir, file)) for file in files])
        for result in results:
            for ptid, info in result.items():
                patients[ptid] = info

    # write output
    with open(os.path.join(args.output_dir, "patients.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "PTID",
                "BIRTH_YR",
                "GENDER",
                "RACE",
                "ETHNICITY",
                "REGION",
                "DIVISION",
                "DECEASED_INDICATOR",
                "DATE_OF_DEATH",
            ]
        )
        for ptid, patient in tqdm(patients.items(), desc="Writing output"):
            writer.writerow(
                [
                    ptid,
                    patient["BIRTH_YR"],
                    patient["GENDER"],
                    patient["RACE"],
                    patient["ETHNICITY"],
                    patient["REGION"],
                    patient["DIVISION"],
                    patient["DECEASED_INDICATOR"],
                    patient["DATE_OF_DEATH"],
                ]
            )


if __name__ == "__main__":
    main()
