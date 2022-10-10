import ujson as json


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, examples):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
