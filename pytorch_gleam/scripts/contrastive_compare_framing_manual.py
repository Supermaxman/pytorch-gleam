import argparse

import pandas as pd

from pytorch_gleam.data.twitter import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    framings = list(read_jsonl(input_path))
    print(f"{len(framings)} framings")
    rows = {}
    for framing in sorted(framings, key=lambda x: x["size"], reverse=True):
        f_id = framing["id"]
        f_size = framing["size"]
        f_clusters = len(framing["docs"])
        fc_id = list(framing["closest_frame"].keys())[0]
        fc_txt = framing["closest_frame"][fc_id]["text"]
        fc_score = framing["closest_frame"][fc_id]["score"]
        rows[f_id] = {
            "tweets": f_size,
            "clusters": f_clusters,
            "text": framing["text"],
            "closest_framing": fc_id,
            "closest_f_score": fc_score,
            "closest_f_text": fc_txt,
        }

    df = pd.DataFrame.from_dict(data=rows, orient="index")
    print("Saving...")
    df.to_excel(output_path)

    print("Done!")


if __name__ == "__main__":
    main()
