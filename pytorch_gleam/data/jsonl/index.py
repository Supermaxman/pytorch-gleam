import os
from contextlib import ContextDecorator

import ujson as json


class JsonlIndex(ContextDecorator):
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

    def get(self, ex_id):
        position = self.index[ex_id]
        self.f.seek(position)
        line = self.f.readline()
        ex = json.loads(line)
        return ex

    def __getitem__(self, ex_id):
        return self.get(ex_id)

    def __len__(self):
        return len(self.index)

    def __iter__(self):
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
            while line := f.readline():
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    ex_id = ex["id"]
                    ex_position = f.tell()
                    index[ex_id] = ex_position

        if index_path is None:
            index_path = path + ".index.json"
        with open(index_path, "w") as f:
            json.dump(index, f)
        return index
