import argparse
import os
import string
from collections import defaultdict
from multiprocessing import Pool

import ujson as json
from tqdm import tqdm


def parse_status(post, min_length=40):
    # Status                   304224
    # post_type: status
    # has a "message" that can be long. message can be missing, not 100% sure when that happens though...
    post["text"] = post.get("message", post.get("description", ""))
    if len(post["text"]) < min_length:
        return None
    return post


def add_punctuation_ending(text):
    if text is not None and text[-1] not in string.punctuation:
        text += "."
    return text


def parse_link(post, min_length=40):
    # Link                    3231519
    # post_type: link is for external links with some text in
    # "title", "caption", "description", and "message"
    # check for overlap between these four things and only include unique text
    # <title> + <caption> + <description> + <message>
    text_fields = ["title", "caption", "description", "message"]
    texts = [add_punctuation_ending(post.get(text)) for text in text_fields]
    text = " \n".join([text for text in texts if text is not None])
    post["text"] = text
    if len(post["text"]) < min_length:
        return None
    return post


def parse_photo(post, min_length=40):
    #
    # Photo                   1502029
    # post_type: photo
    # can still have a "message", but otherwise drop
    #
    text = post.get("description", post.get("message", ""))
    post["text"] = text
    if len(post["text"]) < min_length:
        return None
    return post


def parse_video(post, min_length=40):
    # Native Video             233817
    # post_type: native_video
    # can have description and message, but if shot skip
    # Video                     50846
    # post_type: video
    # can have a message, but may be so short very little is included
    # Live Video Complete       62962
    # post_type: live_video_complete
    # can have message, but if short then skip
    #
    # TODO message could be different and have more content
    text = post.get("description", post.get("message", ""))
    post["text"] = text
    if len(post["text"]) < min_length:
        return None
    return post


def parse_youtube(post, min_length=40):
    # YouTube                   61392
    # post_type: youtube
    # title and message, maybe description
    text_fields = ["title", "description", "message"]
    texts = [add_punctuation_ending(post.get(text)) for text in text_fields]
    text = " \n".join([text for text in texts if text is not None])
    post["text"] = text
    if len(post["text"]) < min_length:
        return None
    return post


post_type_parse = {
    "status": parse_status,
    "link": parse_link,
    "photo": parse_photo,
    "video": parse_video,
    "native_video": parse_video,
    "live_video_complete": parse_video,
    "youtube": parse_youtube,
}


def parse_post(post):
    """Parse a Facebook post.

    Returns `None` if post does not match correct post_type or content requirements, such as minimum length.
    """
    # platformId and platform should become "id"
    # might also already have "id" from CrowdTangle, keep this id if needed
    if "id" in post:
        post["crowdId"] = post["id"]
    post["id"] = post["platform"] + "|" + post["platformId"]
    post_type = post["type"]
    if post_type not in post_type_parse:
        return None

    post = post_type_parse[post_type](post)
    return post


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_paths", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-ps", "--processes", default=8)
    args = parser.parse_args()
    all_ids = set()
    files = []
    for path in args.input_paths.split(","):
        path_files = [(os.path.join(path, x), path) for x in os.listdir(path) if x.endswith(".json")]
        files.extend(path_files)
    path_counts = defaultdict(int)
    with open(args.output_path, "w") as f:
        with Pool(processes=args.processes) as p_parse:
            for posts in tqdm(p_parse.imap_unordered(parse_post_file, files), total=len(files)):
                for post_id, post_json, file_path, path in posts:
                    if post_id in all_ids:
                        continue
                    f.write(post_json + "\n")
                    all_ids.add(post_id)
                    path_counts[path] += 1

    for path, count in path_counts.items():
        print(f"{path} - {count}")


def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def parse_posts(posts):
    for post in posts:
        post = parse_post(post)
        if post is not None:
            yield post


def parse_post_file(args):
    file_path, path = args
    if file_path.endswith(".json"):
        posts = read_json(file_path)
        parsed_posts = []
        for post in parse_posts(posts):
            json_data = json.dumps(post)
            parsed_posts.append((post["id"], json_data, file_path, path))
        return parsed_posts
    else:
        raise ValueError(f"Unknown file format: {file_path}")


if __name__ == "__main__":
    main()
