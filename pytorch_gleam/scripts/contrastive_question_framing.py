import argparse

from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig, write_jsonl


def parse_text(ex_text, preprocess_config):
    doc_txt = preprocess_tweet(ex_text, preprocess_config)
    doc_txt = doc_txt.replace("\n ", " ").replace(" via ", " ")
    doc_txt = doc_txt.replace("twitteruser", " ").replace("twitterurl", " ")
    # TODO filter out hashtags?
    doc_txt = " ".join(doc_txt.split())
    return doc_txt


def preprocess_example(example, preprocess_config):
    ex_id = example["id"]
    # sorted
    all_docs = list(sorted(example["docs"], key=lambda x: len(parse_text(x["text"], preprocess_config)), reverse=True))
    doc = all_docs[0]
    doc_txt = parse_text(doc["text"], preprocess_config)
    ex = {"id": ex_id, "text": doc_txt, "docs": all_docs, "size": sum([len(c["docs"]) for c in example["docs"]])}
    return ex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    preprocess_config = TweetPreprocessConfig(
        asciify_emojis=False, do_lower_case=False, replace_multiple_usernames=False, replace_multiple_urls=False
    )

    print("Loading data...")
    results = [preprocess_example(ex, preprocess_config) for ex in read_jsonl(input_path)]
    print("Saving results...")
    write_jsonl(output_path, results)

    print("Done!")


if __name__ == "__main__":
    main()
