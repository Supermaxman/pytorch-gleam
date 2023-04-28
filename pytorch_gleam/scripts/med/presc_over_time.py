import argparse
import csv
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# Increase field limit to avoid csv.Error: field larger than field limit (131072)
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
csv.field_size_limit(sys.maxsize)


def read_file(file_path: str, patterns: Dict[str, re.Pattern]) -> Dict[str, Dict[str, List[str]]]:
    current = mp.current_process()
    pos = current._identity[0] - 1

    # each .txt file is a csv with | as delimiter
    # the columns are provided for the first row as:
    # PTID|RXDATE|RXTIME|DRUG_NAME|NDC|NDC_SOURCE|PROVID|ROUTE|QUANTITY_OF_DOSE|STRENGTH|STRENGTH_UNIT|DOSAGE_FORM|DAILY_DOSE|DOSE_FREQUENCY|QUANTITY_PER_FILL|NUM_REFILLS|DAYS_SUPPLY|GENERIC_DESC|DRUG_CLASS|DISCONTINUE_REASON|SOURCEID
    drug_date_patients: Dict[str, Dict[str, List[str]]] = {}
    file_name = os.path.basename(file_path).split(".")[0]
    with open(file_path, "r") as f:
        reader = csv.DictReader(f, delimiter="|")
        # estimate from the first file that there are 41,461,047 rows
        for row in tqdm(reader, total=41_461_047, desc=file_name, position=pos, miniters=41_461_047 / 1_000):
            # skip if rxdate is not a valid date
            # dates are in the format MM-DD-YYYY (e.g. 07-14-2021)
            try:
                date = row["RXDATE"]
                date = datetime.strptime(date, "%m-%d-%Y")
            except ValueError:
                continue
            # convert date to string in the format YYYY-MM-DD (e.g. 2021-07-14)
            # skip if rxdate is before 2020-01-01
            if date.year < 2020:
                continue
            date = date.strftime("%Y-%m-%d")
            for drug, pattern in patterns.items():
                if pattern.match(row["DRUG_NAME"]):
                    if drug not in drug_date_patients:
                        drug_date_patients[drug] = {}
                    if date not in drug_date_patients[drug]:
                        drug_date_patients[drug][date] = []
                    drug_date_patients[drug][date].append(row["PTID"])
    return drug_date_patients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/shared/aifiles/disk1/covid19/raw/rawdata/20220120")
    parser.add_argument("--output_dir", type=str, default="/shared/aifiles/disk1/covid19/artifacts/presc_over_time")
    parser.add_argument("--prefix", type=str, default="cov_20220120_rx_presc_")

    args = parser.parse_args()
    patterns = {
        "Hydroxychloroquine": re.compile("(HYDROXYCHLOROQUINE|PLAQUENIL)", re.IGNORECASE),
        "Ivermectin": re.compile("(IVERMECTIN|STROMECTOL)", re.IGNORECASE),
        "Remdesivir": re.compile("(REMDESIVIR)", re.IGNORECASE),
    }
    # get all files in data_dir
    # files are named cov_20220120_rx_presc_0.txt, cov_20220120_rx_presc_1.txt, etc
    files = sorted(
        [file for file in os.listdir(args.data_dir) if file.endswith(".txt") and file.startswith(args.prefix)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # mkdir for output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # read files in parallel
    drug_date_patients: Dict[str, Dict[str, str]] = defaultdict(lambda: defaultdict(list))
    with mp.Pool(processes=len(files), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = pool.starmap(read_file, [(os.path.join(args.data_dir, file), patterns) for file in files])
        for result in results:
            for drug, date_patients in result.items():
                for date, patients in date_patients.items():
                    drug_date_patients[drug][date].extend(patients)

    # write output
    with open(os.path.join(args.output_dir, "presc_patients_over_time.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Drug", "Date", "PTID"])
        for drug, date_patients in tqdm(drug_date_patients.items(), desc="Writing output"):
            for date, patients in date_patients.items():
                for patient in patients:
                    writer.writerow([drug, date, patient])


if __name__ == "__main__":
    main()
