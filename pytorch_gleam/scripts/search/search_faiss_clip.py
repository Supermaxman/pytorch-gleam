import argparse
import os
from pathlib import Path

import clip
import faiss
import numpy as np
import pandas as pd
import torch
import ujson as json
from tqdm.rich import tqdm


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, data):
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")


def embed_text(model, device, text: str):
    text_tokens = clip.tokenize([text], truncate=True)

    text_features = model.encode_text(text_tokens.to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_embeddings = text_features.cpu().detach().numpy().astype(np.float32)
    return text_embeddings


def query_index(index, model, device, text: str, image_list, k: int):
    text_embeddings = embed_text(model, device, text)
    distances, indices = index.search(text_embeddings, k=k)
    results = []
    for rank, (d, i) in enumerate(zip(distances[0], indices[0]), start=1):
        file_name = image_list[i]
        results.append(
            {
                "rank": rank,
                "distance": float(d),
                "file_name": file_name,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B/32",
        help="CLIP model name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top k images to return",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="/shared/aifiles/disk1/media/twitter/v10/covid19-twitter-images-dedup-emb-vit-b-32-index/image.index",
        help="Path to index",
    )
    parser.add_argument(
        "--query_path",
        type=str,
        default="/users/max/data/corpora/co-vax-frames/covid19/co-vax-frames.json",
        help="Path to query file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/shared/aifiles/disk1/media/twitter/v10/cti-dedup-vit-b-32-frames-results.json",
        help="Path to output file",
    )

    args = parser.parse_args()
    model_name = args.model_name
    top_k = args.top_k
    index_path = args.index_path
    query_path = args.query_path
    output_path = args.output_path

    print("Loading queries...")
    with open(query_path) as f:
        queries = json.load(f)
    print(f"  Total queries: {len(queries)}")

    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    print(f'  CLIP model loaded on "{device}"')

    print("Loading image list...")
    meta_path = os.path.join(os.path.dirname(index_path), "metadata")
    data_dir = Path(meta_path)
    df = pd.concat(pd.read_parquet(parquet_file) for parquet_file in data_dir.glob("*.parquet"))
    image_list = df["image_path"].tolist()
    print(f"  Total images: {len(image_list)}")

    print("Loading index...")
    index = faiss.read_index(index_path)

    print("Querying index...")
    total = 0
    results = {}
    for q_id, q in tqdm(queries.items()):
        query = q["text"]
        q_results = query_index(index, model, device, query, image_list, top_k)
        results[q_id] = q_results
        total += len(q_results)
    print(f"  Total images found: {total}")
    print("Writing results...")
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"  Results written to {output_path}")
    print("DONE!")


if __name__ == "__main__":
    main()
