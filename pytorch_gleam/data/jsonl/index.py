import os
from contextlib import ContextDecorator
from typing import Any, BinaryIO, Dict, Iterator, List, Union

import ujson as json


class JsonlIndex(ContextDecorator):
    index: Dict[str, int]
    f: Union[None, BinaryIO]
    path: str
    index_path: str

    def __init__(self, path: str, index_path: str = None):
        self.path = path
        if index_path is None:
            index_path = path + ".index.json"
        self.index_path = index_path
        if not os.path.exists(index_path):
            print("No index found, creating...")
            self.index = self.create(path, index_path)
            print("Index created")
        else:
            with open(index_path, "r") as f:
                self.index = json.load(f)
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "rb")
        return self

    def __exit__(self, *exc):
        self.f.close()
        return False

    def get(self, ex_id: Union[str, List[str]]) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        if isinstance(ex_id, str):
            return self._get_one(ex_id)
        else:
            # seek in order from lowest to highest byte position
            sorted_ex_ids = sorted(self.index[x_id] for x_id in ex_id)
            for ex_pos in sorted_ex_ids:
                yield self._get_pos(ex_pos)

    def _get_one(self, ex_id) -> Dict[str, Any]:
        position = self.index[ex_id]
        return self._get_pos(position)

    def _get_pos(self, ex_position: int) -> Dict[str, Any]:
        self.f.seek(ex_position)
        line = self.f.readline()
        ex = json.loads(line)
        return ex

    def __getitem__(self, ex_id: Union[str, List[str]]):
        return self.get(ex_id)

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.f.seek(0)
        while line := self.f.readline():
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex

    @staticmethod
    def write(path, examples, index_path=None):
        index = {}
        with open(path, "wb") as f:
            for ex in examples:
                ex_id = ex["id"]
                ex_position = f.tell()
                index[ex_id] = ex_position
                f.write(json.dumps(ex) + "\n")

        if index_path is None:
            index_path = path + ".index.json"
        with open(index_path, "w") as f:
            json.dump(index, f)

    @staticmethod
    def create(path: str, index_path: str = None):
        index = {}
        with open(path, "rb") as f:
            # must be done for f.tell to work inside for loop over lines
            while True:
                # needs to happen BEFORE readline to get starting position
                ex_position = f.tell()
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    ex_id = ex["id"]
                    index[ex_id] = ex_position

        if index_path is None:
            index_path = path + ".index.json"
        with open(index_path, "w") as f:
            json.dump(index, f)
        return index
