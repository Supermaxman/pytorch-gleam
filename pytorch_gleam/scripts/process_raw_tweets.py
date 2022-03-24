import argparse
import os
from collections import defaultdict
from multiprocessing import Pool
from pprint import pprint

import ujson as json
from tqdm import tqdm


def read_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def invert_errors(errors):
    inv = defaultdict(dict)
    for error in errors:
        r_id = error["resource_id"]
        e_type = error["detail"].replace(f": [{r_id}].", "")
        inv[r_id][e_type] = error
    return inv


def invert_ids(items):
    inv = {}
    for item in items:
        item_id = item["id"] if "id" in item else item["media_key"]
        inv[item_id] = item
        if "username" in item:
            inv[item["username"]] = item
    return inv


def invert_includes(includes):
    inv = {}
    for key, vals in includes.items():
        inv[key] = invert_ids(vals)
    return inv


def parse_tweet(tweet, inv_includes, inv_errors):
    author_id = tweet["author_id"]
    if author_id in inv_errors:
        author = inv_errors[author_id]
    else:
        author = inv_includes["users"][author_id]
    tweet["author"] = author
    if "entities" not in tweet:
        tweet["entities"] = {}
    for e_type, e_vals in tweet["entities"].items():
        if e_type == "mentions":
            for e in e_vals:
                if "username" in e:
                    e_username = e["username"]
                    if e_username in inv_errors:
                        e_user = inv_errors[e_username]
                    elif e_username in inv_includes["users"]:
                        e_user = inv_includes["users"][e_username]
                    else:
                        e_user = None
                    e["user"] = e_user
    if "referenced_tweets" not in tweet:
        tweet["referenced_tweets"] = []
    is_retweet = False
    for ref_tweet in tweet["referenced_tweets"]:
        r_id = ref_tweet["id"]
        if r_id in inv_errors:
            r_tweet = inv_errors[r_id]
        elif r_id in inv_includes["tweets"]:
            r_tweet = inv_includes["tweets"][r_id]
        else:
            r_tweet = None
        ref_tweet["data"] = r_tweet
        if ref_tweet["type"] == "retweeted":
            is_retweet = True
    return tweet, is_retweet


def parse_tweets(tweets, keep_retweets: bool):
    if "data" not in tweets:
        return
    t_data = tweets["data"]
    if "includes" not in tweets:
        tweets["includes"] = {}
    if "errors" not in tweets:
        tweets["errors"] = []
    t_includes = invert_includes(tweets["includes"])
    t_errors = invert_errors(tweets["errors"])
    for tweet in t_data:
        try:
            tweet, is_retweet = parse_tweet(tweet, t_includes, t_errors)
            if not is_retweet or (is_retweet and keep_retweets):
                yield tweet
        except Exception as e:
            pprint(e)
            pprint(tweet)


def parse_tweet_file(file_path, keep_retweets):
    tweets = read_file(file_path)
    parsed_tweets = []
    for tweet in parse_tweets(tweets, keep_retweets):
        json_data = json.dumps(tweet)
        parsed_tweets.append((tweet["id"], json_data))
    return parsed_tweets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_paths", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-ps", "--processes", default=8)
    parser.add_argument("-rt", "--retweets", action="store_true")
    args = parser.parse_args()
    files = []
    for path in args.input_path.split(","):
        path_files = [(os.path.join(path, x), args.retweets) for x in os.listdir(path) if x.endswith(".json")]
        files.extend(path_files)

    all_ids = set()
    with open(args.output_path, "w") as f:
        with Pool(processes=args.processes) as p:
            for tweets in tqdm(p.imap(parse_tweet_file, files), total=len(files)):
                for tweet_id, tweet_json in tweets:
                    if tweet_id in all_ids:
                        continue
                    f.write(tweet_json + "\n")
                    all_ids.add(tweet_id)


if __name__ == "__main__":
    main()
