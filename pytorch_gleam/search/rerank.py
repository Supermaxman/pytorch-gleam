import argparse
import json
import logging
import os

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, BertTokenizer


def get_tweet_text(tweet):
    tweet_text = tweet["text"] if "text" in tweet else tweet["full_text"]
    if tweet_text.startswith("RT "):
        ref_tweets = tweet["referenced_tweets"]
        if len(ref_tweets) > 0:
            rt_data = ref_tweets[0]["data"]
            if "text" in rt_data:
                tweet_text = rt_data["text"]
    if "entities" in tweet:
        for e_type, e_list in tweet["entities"].items():
            if e_type == "urls":
                for e_url in e_list:
                    r_url = e_url["url"]
                    s_url = e_url["expanded_url"]
                    tweet_text = tweet_text.replace(r_url, s_url)
    return tweet_text


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ex = json.loads(line)
                    yield ex
                except Exception as e:
                    print(e)


def get_tweets(dir_path):
    for file_name in sorted(list(os.listdir(dir_path))):
        file_path = os.path.join(dir_path, file_name)
        for ex in read_jsonl(file_path):
            yield ex


def worker_init_fn(_):
    try:
        process_id = dist.get_rank()
        num_processes = dist.get_world_size()
    except Exception:
        process_id = 0
        num_processes = 1
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    print(f"INFO: WORKER_INIT WORKER_INFO: {worker_id}/{num_workers}")
    print(f"INFO: WORKER_INIT: RANK_INFO: {process_id}/{num_processes}")
    dataset = worker_info.dataset
    dataset.frequency = (process_id * num_workers) + worker_id
    dataset.num_workers = num_processes * num_workers
    print(f"INFO: WORKER_INIT: F_INFO: {dataset.frequency}/{dataset.num_workers}")


class RerankDataset(IterableDataset):
    def __init__(self, index_path, scores_path, questions_path, worker_estimate=6):
        with open(questions_path, "r") as f:
            self.questions = json.load(f)
        with open(scores_path, "r") as f:
            scores = json.load(f)

        self.index_path = index_path
        self.frequency = 0
        self.num_workers = 1
        self.tweet_examples = scores
        self.num_examples = 0
        self.worker_estimate = worker_estimate
        self.num_examples = sum(len(q_scores) for q_scores in scores.values())

        print(f"Num examples: {self.num_examples}")

    def __len__(self):
        return self.num_examples // self.worker_estimate

    def __iter__(self):
        ex_idx = 0
        for tweet in get_tweets(self.index_path):
            tweet_id = tweet["id"]
            if tweet_id not in self.tweet_examples:
                continue
            tweet_text = tweet["contents"]
            # only do relevance re-ranking on bm25 results
            q_exs = self.tweet_examples[tweet_id]
            # OR do relevance re-ranking on all results (EXPENSIVE)
            # q_exs = self.questions
            for q_id in q_exs:
                ex = {
                    "id": tweet_id,
                    "question_id": f"{q_id}",
                    "query": self.questions[q_id]["text"],
                    "text": tweet_text,
                }
                if ex_idx % self.num_workers == self.frequency:
                    yield ex
                ex_idx += 1


class RerankBatchCollator(object):
    def __init__(self, tokenizer, max_seq_len: int, force_max_seq_len: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.force_max_seq_len = force_max_seq_len

    def __call__(self, examples):
        ids = []
        question_ids = []
        sequences = []
        for ex in examples:
            ids.append(ex["id"])
            question_ids.append(ex["question_id"])
            sequences.append((ex["query"], ex["text"]))

        tokenizer_batch = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=sequences,
            add_special_tokens=True,
            padding="max_length" if self.force_max_seq_len else "longest",
            return_tensors="pt",
            truncation="only_second",
            max_length=self.max_seq_len,
        )
        batch = {
            "id": ids,
            "question_id": question_ids,
            "input_ids": tokenizer_batch["input_ids"],
            "attention_mask": tokenizer_batch["attention_mask"],
            "token_type_ids": tokenizer_batch["token_type_ids"],
        }

        return batch


def get_device_id():
    try:
        device_id = dist.get_rank()
    except Exception:
        if "XRT_SHARD_ORDINAL" in os.environ:
            device_id = int(os.environ["XRT_SHARD_ORDINAL"])
        else:
            device_id = 0
    return device_id


class RerankBert(pl.LightningModule):
    def __init__(self, pre_model_name, predict_path=None):
        super().__init__()
        self.pre_model_name = pre_model_name
        self.predict_path = predict_path
        os.makedirs(self.predict_path, exist_ok=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(pre_model_name)
        self.config = self.bert.config
        self.save_hyperparameters()
        self.file = None

    def forward(self, input_ids, attention_mask, token_type_ids):
        # [batch_size, 2]
        logits = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        return logits

    def training_step(self, batch, batch_nb):
        pass

    def test_step(self, batch, batch_nb):
        return self._eval_step(batch, batch_nb, "test")

    def validation_step(self, batch, batch_nb):
        return self._eval_step(batch, batch_nb, "val")

    def _forward_step(self, batch, batch_nb):
        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )

        return logits

    def _eval_step(self, batch, batch_nb, name):
        logits = self._forward_step(batch, batch_nb)
        logits = logits.detach().cpu()
        device_id = get_device_id()
        if self.file is None:
            self.file = open(os.path.join(self.predict_path, f"predictions-{device_id}.jsonl"), "w")
        for i in range(len(logits)):
            self.file.write(
                json.dumps(
                    {
                        "id": batch["id"][i],
                        "question_id": batch["question_id"][i],
                        "pos_score": logits[i, 1].item(),
                        "neg_score": logits[i, 0].item(),
                    }
                )
                + "\n"
            )

    def _eval_epoch_end(self, name):
        if self.file is not None:
            self.file.close()
            self.file = None

    def on_validation_epoch_end(self, trainer, pl_module):
        self._eval_epoch_end("val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._eval_epoch_end("test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index_path", required=True)
    parser.add_argument("-qp", "--questions_path", default=None)
    parser.add_argument("-sp", "--scores_path", default=None)
    parser.add_argument("-op", "--output_path", required=True)
    parser.add_argument("-pm", "--pre_model_name", default="nboost/pt-biobert-base-msmarco")
    parser.add_argument("-sd", "--save_directory", default="models")
    parser.add_argument("-bs", "--batch_size", default=4, type=int)
    parser.add_argument("-w", "--num_workers", default=1, type=int)
    parser.add_argument("-ml", "--max_seq_len", default=96, type=int)
    parser.add_argument("-se", "--seed", default=0, type=int)
    parser.add_argument("-cd", "--torch_cache_dir", default=None)
    parser.add_argument("-gpu", "--gpus", default="0")
    parser.add_argument("-ts", "--train_sampling", default="none")
    parser.add_argument("-ls", "--losses", default="compare_loss")

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # export TPU_IP_ADDRESS=10.155.6.34
    # export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    gpus = [int(x) for x in args.gpus.split(",")]
    # precision = 32
    num_workers = args.num_workers
    deterministic = True

    tokenizer = BertTokenizer.from_pretrained(args.pre_model_name)

    logging.info("Loading datasets...")

    val_dataset = RerankDataset(
        index_path=args.index_path,
        scores_path=args.scores_path,
        questions_path=args.questions_path,
        worker_estimate=len(gpus),
    )
    val_data_loader = DataLoader(
        val_dataset,
        num_workers=num_workers,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=RerankBatchCollator(
            tokenizer,
            args.max_seq_len,
        ),
        worker_init_fn=worker_init_fn,
    )

    logging.info("Loading model...")

    model = RerankBert(
        pre_model_name=args.pre_model_name,
        predict_path=args.output_path,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=0,
        deterministic=deterministic,
    )

    logging.info("Predicting...")
    try:
        trainer.test(model, val_data_loader)
    except Exception as e:
        logging.exception("Exception during predicting", exc_info=e)


if __name__ == "__main__":
    main()
