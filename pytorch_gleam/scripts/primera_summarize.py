import argparse

import pytorch_lightning as pl
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LEDForConditionalGeneration

from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig, write_jsonl


def preprocess_example(example, preprocess_config):
    all_docs = []
    ex_id = example["id"]
    for doc in example["docs"]:
        doc_txt = preprocess_tweet(doc["text"], preprocess_config)
        doc_txt = doc_txt.replace("\n ", " ").replace(" via ", " ")
        doc_txt = doc_txt.replace("twitteruser", " ").replace("twitterurl", " ")
        doc_txt = " ".join(doc_txt.split())
        all_docs.append(doc_txt)

    ex = {"ids": ex_id, "documents": all_docs}
    return ex


def run_model(example, device, model, tokenizer, max_input_len=4096, **generate_args):
    doc_sep_token_id = tokenizer.convert_tokens_to_ids("<doc-sep>")
    # we will tokenize a single example document,
    # and we will move these tensors to the GPU device:
    input_ids = []
    for doc in example["documents"]:
        input_ids.extend(
            tokenizer.encode(
                doc,
                truncation=True,
                max_length=max_input_len // len(example["documents"]),
            )[1:-1]
        )
        input_ids.append(doc_sep_token_id)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

    input_ids = torch.tensor([input_ids]).to(device)

    global_attention_mask = torch.zeros_like(input_ids).to(input_ids.device)
    global_attention_mask[:, 0] = 1
    global_attention_mask[input_ids == doc_sep_token_id] = 1

    # the outputs will contain decoded token ids
    # based on the estimated most likely summary sequence
    # using various decoding options
    multi_summary_ids = model.generate(
        input_ids=input_ids,
        global_attention_mask=global_attention_mask,
        use_cache=True,
        **generate_args,
    )[:, 1:]
    # converts token ids back to strings for multiple summaries
    summaries = tokenizer.batch_decode(multi_summary_ids, skip_special_tokens=True)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-m", "--model_name", default="allenai/PRIMERA")
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    seed = args.seed

    pl.seed_everything(seed)

    preprocess_config = TweetPreprocessConfig()

    print("CUDA Enabled: ", torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"  {device} - " + torch.cuda.get_device_name(0))
    else:
        print(f"  {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = LEDForConditionalGeneration.from_pretrained(model_name)

    print(f"Loading model on {device}...")
    # move model to GPU device
    model.to(device)
    # turn on EVAL mode so drop-out layers do not randomize outputs
    model.eval()

    print("Loading data...")
    examples = [preprocess_example(ex, preprocess_config) for ex in read_jsonl(input_path)]
    results = []
    print("Running model on data...")
    for ex in tqdm(examples):
        summaries = run_model(
            ex,
            device,
            model,
            tokenizer,
            max_input_len=1024,
            max_length=32,
            min_length=5,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            num_beams=10,
            num_return_sequences=1,
            early_stopping=True,
        )
        ex_id = ex["ids"]

        summary = summaries[0]
        result = {"ids": ex_id, "summary": summary}
        results.append(result)
    print("Saving results...")
    write_jsonl(output_path, results)

    print("Done!")


if __name__ == "__main__":
    main()
