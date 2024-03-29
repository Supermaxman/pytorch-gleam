import argparse
import os
from collections import defaultdict
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import ujson as json
from pytorch_lightning.callbacks import BasePredictionWriter


class JsonlWriter(BasePredictionWriter):
    def __init__(self, write_interval: str = "batch", output_path: str = None):
        super().__init__(write_interval)
        self.output_path = output_path
        self.first_write = False
        self.file = None

    def _write(self, trainer, prediction):
        if self.output_path is None:
            predictions_dir = os.path.join(trainer.default_root_dir, "predictions")
        else:
            predictions_dir = self.output_path

        os.makedirs(predictions_dir, exist_ok=True)

        try:
            process_id = dist.get_rank()
        except RuntimeError:
            process_id = 0
        file_name = f"predictions-{process_id}.jsonl"

        pred_file = os.path.join(predictions_dir, file_name)
        rows = defaultdict(dict)
        for key, values in prediction.items():
            for ex_idx, ex_value in enumerate(values):
                if isinstance(ex_value, torch.Tensor):
                    ex_value = ex_value.tolist()
                rows[ex_idx][key] = ex_value
        rows = rows.values()
        if self.file is None:
            self.file = open(pred_file, "w")
        for row in rows:
            self.file.write(json.dumps(row) + "\n")

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        self._write(trainer, prediction)

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        for prediction in predictions:
            self._write(trainer, prediction)

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.file is not None:
            self.file.close()


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def read_predictions(input_path):
    predictions = defaultdict(dict)
    total_count = 0
    unique_set = set()
    for file_name in sorted(os.listdir(input_path)):
        if file_name.endswith(".jsonl"):
            print(f"{file_name}: ", end="")
            file_path = os.path.join(input_path, file_name)
            f_count = 0
            for pred in read_jsonl(file_path):
                tweet_id, f_id = pred["ids"].split("|")
                unique_set.add((tweet_id, f_id))
                scores = pred["scores"]
                predictions[tweet_id][f_id] = scores
                f_count += 1
            print(f"{f_count}")
            total_count += f_count
    print(f"TOTAL: {total_count}")
    print(f"UNIQUE: {len(unique_set)}")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()
    predictions = read_predictions(args.input_path)
    with open(args.output_path, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
