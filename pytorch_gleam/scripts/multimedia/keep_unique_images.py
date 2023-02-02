import os
import shutil
from collections import defaultdict

import ujson as json
from tqdm import tqdm

enc_path = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-encs.jsonl"
dupe_path = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dupes.json"
img_folder = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images"
new_img_folder = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dedup"


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


encodings = {}
print("Loading encodings")
for ex in tqdm(read_jsonl(enc_path), total=2_200_000):
    for file, file_hash in ex.items():
        encodings[file] = file_hash


print("Calculating hash dupes")
inv_index = defaultdict(list)
for file, file_hash in tqdm(encodings.items()):
    inv_index[file_hash].append(file)

print("Creating dupe lists")
duplicates = {}
for file_hash, hash_files in tqdm(inv_index.items()):
    for file in hash_files:
        new_dupes = [f for f in hash_files if f != file]
        duplicates[file] = new_dupes

print("Moving non-duplicate files")
dup_files = set()
for file, file_dupes in tqdm(sorted(duplicates.items(), key=lambda x: x[0])):
    if file in dup_files:
        continue
    shutil.copy(os.path.join(img_folder, file), new_img_folder)
    for dup_file in file_dupes:
        dup_files.add(dup_file)
print("Done!")
