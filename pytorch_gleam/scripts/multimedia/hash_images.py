import os
from multiprocessing import Pool

import ujson as json
from imagededup.methods import PHash
from tqdm import tqdm

img_folder = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images"
dupe_path = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dupes.json"
enc_path = "/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-encs.jsonl"

phasher = PHash()


def encode(file):
    phasher = PHash()
    file_path = os.path.join(img_folder, file)
    file_hash = phasher.encode_image(file_path)
    return file, file_hash


if __name__ == "__main__":
    encodings = {}
    with Pool(processes=8) as pool:
        with open(enc_path, "w") as f:
            print("Encoding images")
            with os.scandir(img_folder) as it:
                for file, file_hash in tqdm(pool.imap_unordered(encode, (x.name for x in it)), total=2_200_000):
                    encodings[file] = file_hash
                    f.write(json.dumps({file: file_hash}) + "\n")
