import argparse

import pytorch_lightning as pl
import torch
from longformer import LongformerEncoderDecoderConfig, LongformerEncoderDecoderForConditionalGeneration
from longformer.sliding_chunks import pad_to_window_size
from tqdm import tqdm
from transformers import AutoTokenizer

from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig, write_jsonl


def preprocess_example(example, preprocess_config):
    all_docs = []
    ex_id = example["id"]
    for doc in example["docs"]:
        doc_txt = preprocess_tweet(doc["text"], preprocess_config)
        all_docs.append(doc_txt)

    ex = {"ids": ex_id, "documents": all_docs}
    return ex


# Re-run this cell when you swap models
def run_model(example, device, model, tokenizer, max_input_len=4096, max_output_len=1024, **generate_args):
    docsep_token_id = tokenizer.additional_special_tokens_ids[0]
    pad_token_id = tokenizer.pad_token_id
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
        input_ids.append(docsep_token_id)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]

    input_ids = torch.tensor([input_ids]).to(device)

    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    attention_mask[input_ids == pad_token_id] = 0
    # global attention on one token for all model params to be used,
    # which is important for gradient checkpointing to work
    attention_mask[:, 0] = 2
    attention_mask[input_ids == docsep_token_id] = 2
    # attention_mode == "sliding_chunks":
    half_padding_mod = model.config.attention_window[0]

    input_ids, attention_mask = pad_to_window_size(
        # ideally, should be moved inside the LongformerModel
        input_ids,
        attention_mask,
        half_padding_mod,
        pad_token_id,
    )
    # the outputs will contain decoded token ids
    # based on the estimated most likely summary sequence
    # using various decoding options
    multi_summary_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        max_length=max_output_len,
        min_length=0,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        **generate_args,
    )[:, 1:]
    # converts token ids back to strings for multiple summaries
    summaries = tokenizer.batch_decode(multi_summary_ids, skip_special_tokens=True)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-m", "--model_name", default="/users/max/data/models/PRIMER_multixscience")
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = LongformerEncoderDecoderConfig.from_pretrained(model_name)
    model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(model_name, config=config)

    # move model to GPU device
    model.to(device)
    # turn on EVAL mode so drop-out layers do not randomize outputs
    model.eval()

    examples = [preprocess_example(ex, preprocess_config) for ex in read_jsonl(input_path)]
    results = []
    for ex in tqdm(examples):
        summaries = run_model(
            ex,
            device,
            model,
            tokenizer,
            max_input_len=1024,
            max_output_len=128,
            num_beams=10,
            num_return_sequences=1,
            early_stopping=True,
        )
        ex_id = ex["ids"]

        summary = summaries[0]
        result = {"ids": ex_id, "summary": summary}
        results.append(result)
    write_jsonl(output_path, results)


if __name__ == "__main__":
    main()
