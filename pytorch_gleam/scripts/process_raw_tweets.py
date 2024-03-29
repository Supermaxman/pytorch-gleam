import argparse
import itertools
import os
from collections import defaultdict
from multiprocessing import Pool
from pprint import pprint

import ujson as json
from tqdm.rich import tqdm


def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


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


def invert_errors(errors):
    inv = defaultdict(dict)
    for error in errors:
        r_id = error["resource_id"] if "resource_id" in error else error["value"]
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

    users = {}
    if "users" in inv_includes:
        users = inv_includes["users"]
    tweets = {}
    if "tweets" in inv_includes:
        tweets = inv_includes["tweets"]
    # media key => media dict
    media = {}
    if "media" in inv_includes:
        media = inv_includes["media"]
    # id => place dict
    places = {}
    if "places" in inv_includes:
        places = inv_includes["places"]
    # get media from inv_includes
    if author_id in inv_errors:
        author = inv_errors[author_id]
    else:
        author = users[author_id]
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
                    elif e_username in users:
                        e_user = users[e_username]
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
        elif r_id in tweets:
            r_tweet = tweets[r_id]
        else:
            r_tweet = None
        ref_tweet["data"] = r_tweet
        if ref_tweet["type"] == "retweeted":
            is_retweet = True
    if "attachments" not in tweet:
        tweet["attachments"] = {}
    tweet["media"] = {}
    for a_type, a_vals in tweet["attachments"].items():
        if a_type == "media_keys":
            for m_key in a_vals:
                if m_key in media:
                    tweet["media"][m_key] = media[m_key]
    if "geo" not in tweet:
        tweet["geo"] = {}
    tweet["places"] = {}
    for g_type, g_id in tweet["geo"].items():
        if g_type == "place_id":
            if g_id in places:
                tweet["places"][g_id] = places[g_id]
    return tweet, is_retweet


def parse_historical_tweets(tweets, keep_retweets: bool):
    if "data" not in tweets:
        print(f"Missing data: {tweets}")
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


def parse_stream_tweet(tweet, keep_retweets: bool):
    if "data" not in tweet:
        return
    t_data = tweet["data"]
    if "includes" not in tweet:
        tweet["includes"] = {}
    if "errors" not in tweet:
        tweet["errors"] = []
    t_includes = invert_includes(tweet["includes"])
    t_errors = invert_errors(tweet["errors"])
    try:
        tweet, is_retweet = parse_tweet(t_data, t_includes, t_errors)
        if not is_retweet or (is_retweet and keep_retweets):
            return tweet
        return None
    except Exception as e:
        pprint(e)
        pprint(tweet)


def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def parse_tweet_file(args):
    file_path, path, keep_retweets = args
    if file_path.endswith(".json"):
        tweets = read_json(file_path)
        parsed_tweets = []
        for tweet in parse_historical_tweets(tweets, keep_retweets):
            json_data = json.dumps(tweet)
            parsed_tweets.append((tweet["id"], json_data, file_path, path))
        return parsed_tweets
    elif file_path.endswith(".jsonl"):
        parsed_tweets = []
        for tweet in read_jsonl(file_path):
            tweet = parse_stream_tweet(tweet, keep_retweets)
            if tweet is not None:
                json_data = json.dumps(tweet)
                parsed_tweets.append((tweet["id"], json_data, file_path, path))
        return parsed_tweets
    else:
        raise ValueError(f"Unknown file format: {file_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_paths", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-ps", "--processes", default=8)
    parser.add_argument("-rt", "--retweets", action="store_true")
    args = parser.parse_args()
    all_ids = set()
    files = []
    for path in args.input_paths.split(","):
        path_files = [
            (os.path.join(path, x), path, args.retweets)
            for x in os.listdir(path)
            if (x.endswith(".json") or x.endswith(".jsonl"))
        ]
        files.extend(path_files)
    path_counts = defaultdict(int)
    with open(args.output_path, "w") as f:
        with Pool(processes=args.processes) as p_parse:
            for tweets in tqdm(p_parse.imap_unordered(parse_tweet_file, files), total=len(files)):
                for tweet_id, tweet_json, file_path, path in tweets:
                    if tweet_id in all_ids:
                        continue
                    f.write(tweet_json + "\n")
                    all_ids.add(tweet_id)
                    path_counts[path] += 1

    for path, count in path_counts.items():
        print(f"{path} - {count}")


if __name__ == "__main__":
    main()
