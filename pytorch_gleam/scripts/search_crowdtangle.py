import argparse
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import requests
from facebook_scraper import get_posts, set_user_agent
from tqdm import tqdm

set_user_agent("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)")


def format_parameters(endpoint, parameters):
    p_text = "".join(
        [("?" if idx == 0 else "&") + f"{key}={value}" for idx, (key, value) in enumerate(parameters.items())]
    )
    return endpoint + p_text


def datetime_range(start: datetime, end: datetime, step: timedelta):
    end -= step
    while start <= end:
        yield start, start + step
        start += step


def convert_datetime(post):
    for key, value in list(post.items()):
        if isinstance(value, datetime):
            post[key] = value.isoformat()
    return post


def random_delay(scale: float = 1.0, loc: float = 0.0, threshold: float = 2.0):
    normal = np.random.normal(loc=0, scale=1.0)
    normal = np.clip(normal, -threshold, threshold)
    # 0 to threshold ** 2
    # 0 to 4 for threshold = 2
    d = normal**2
    # 0 to 4 * scale + loc
    d_scaled = loc + scale * d
    return d_scaled


def parse_timedelta(ts_str: str):
    p = [int(x) for x in ts_str.split(":")]
    hours = 0
    minutes = 0
    seconds = p[-1]
    if len(p) > 1:
        minutes = p[-2]
        if len(p) > 2:
            hours = p[-3]
            assert len(p) == 3
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def download_facebook_extra(post, media_delay, retry_attempts=3):
    retry_count = 0
    while retry_count < retry_attempts:
        # 1 to 3 seconds + small random delay between repeat requests from 0 to 1 second
        delay = media_delay + random_delay(scale=0.25)
        try:
            post_extra = list(get_posts(post_urls=[post["postUrl"]], cookies="private/cookies.txt"))
            post = convert_datetime(post_extra[0])
        except Exception as e:
            error = str(e).lower().strip()
            if "not found" in error:
                # post was deleted or made private, skip
                break
            print(f'{e}: {post["postUrl"]}')
            if "temporarily blocked" in error:
                # wait 33 minutes
                time.sleep(3600 + 10 * delay)
                continue
            time.sleep(10 * delay)
            retry_count += 1
            continue
        time.sleep(delay)
        return post
    return None


def download_image(media_id, media_url, media_output_path, media_delay, retry_attempts=3):
    media_path = os.path.join(media_output_path, media_id)
    retry_count = 0
    while retry_count < retry_attempts and not os.path.exists(media_path):
        # small random delay from 0.5 to 1 seconds + small random delay between repeat requests from 0 to 1 second
        delay = media_delay + random_delay(scale=0.25)
        try:
            response = requests.get(media_url, allow_redirects=True)
        except Exception as e:
            print(f"{e}: {media_url}")
            time.sleep(10 * delay)
            retry_count += 1
            continue
        status = response.status_code
        if status != 200:
            print(f"{status} {response.reason}: {response.content} for {media_url}")
            if status == 403:
                # no longer available, skip
                break
            time.sleep(10 * delay)
            retry_count += 1
            continue
        content_type = response.headers.get("content-type")
        content_types = content_type.split("/")
        if len(content_types) != 2:
            print(f"Unknown content type: {content_type} for {media_url}")
            break
        if content_types[0] != "image":
            print(f"Not an image: {content_type} for {media_url}")
            break
        image_type = content_types[1]
        media_path = f"{media_path}.{image_type}"
        with open(media_path, "wb") as f:
            f.write(response.content)
        time.sleep(delay)
    return media_path


def download_media(posts, media_output_path, media_delay, platform, retry_attempts=3):
    for post in posts:
        # 1 to 2 seconds + random delay from 0 to 1 seconds
        post_delay = media_delay + random_delay(scale=0.25)
        if "media" not in post:
            return post
        post_id = post["platformId"]
        if platform == "facebook":
            # urls turned out to be stale and not working, so we download the image from crawling
            post["extra"] = download_facebook_extra(post, post_delay, retry_attempts)
            extra = post["extra"]
            if extra is None:
                return post
            if "images" not in extra:
                return post
            images = extra["images"]
            # TODO consider videos
            if images is None:
                return post
            if len(images) == 0:
                return post
            for image_idx, media_url in enumerate(images):
                media_id = f"{post_id}_image_{image_idx}"
                # small random delay from 0.5 to 1 seconds
                image_delay = random_delay(scale=0.125, loc=0.5)
                download_image(media_id, media_url, media_output_path, image_delay, retry_attempts)
        else:
            raise NotImplementedError(f"Platform {platform} media not implemented")


def main():
    parser = argparse.ArgumentParser()
    # output_path = "/users/max/data/corpora/covid19-vaccine-facebook/raw-v3"
    # v1
    # start_date = date(2020, 1, 1)
    # end_date = date(2022, 1, 1)
    # v2
    # start_date_str = "2020-12-14T13:00:00"
    # start_date_str = "2020-12-15T21:00:00"
    # end_date_str = "2022-01-01T00:00:00"
    # request_time_delta_str = "01:00:00"
    # 50 calls per minute * 1 minute / 60 seconds = 0.8333 calls per second
    covid_terms = [
        "covid",
        "coronavirus",
        "corona",
        "covid-19",
        "covid19",
        "SARS-CoV-2",
        "SARS",
        "SARS-CoV",
    ]
    covid_query = " OR ".join(covid_terms)
    vaccine_terms = [
        "vaccine",
        "vaccines",
        "vaccination",
        "vaccinations",
        "vax",
        "vaxx",
        "vaxxed",
        "jab",
        "jabbed",
        "vaccinate",
        "vaccinated",
        "vaccinates",
    ]
    vaccine_query = " OR ".join(vaccine_terms)
    default_search_query = f"({covid_query}) AND ({vaccine_query})"

    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-mo", "--media_output_path", required=True)
    parser.add_argument("-s", "--start", required=True)
    parser.add_argument("-e", "--end", required=True)
    parser.add_argument("-q", "--query", default=default_search_query)
    parser.add_argument("-u", "--endpoint_url", default="https://api.crowdtangle.com/posts/search")
    parser.add_argument("-p", "--platform", default="facebook")
    parser.add_argument("-ln", "--language", default="en")
    parser.add_argument("-rd", "--request_delay", type=float, default=50.0 / 60.0)
    parser.add_argument("-md", "--media_delay", type=float, default=1.0)
    parser.add_argument("-rc", "--request_max_count", type=int, default=100)
    parser.add_argument("-sp", "--secrets_path", default="private/secrets.json")
    parser.add_argument("-st", "--secrets_type", default="crowdtangle")
    parser.add_argument("-td", "--request_time_delta", default="01:00:00", help="hours:minutes:seconds")
    parser.add_argument("-qtf", "--query_time_format", default="%Y-%m-%dT%H:%M:%S")
    parser.add_argument("-ftf", "--file_time_format", default="%Y%m%dT%H%M%S")
    args = parser.parse_args()

    start_date_str = args.start
    end_date_str = args.end
    output_path = args.output_path
    media_output_path = args.media_output_path
    search_query = args.query
    language = args.language
    request_time_delta_str = args.request_time_delta
    query_time_format = args.query_time_format
    file_time_format = args.file_time_format
    q_delay = args.request_delay
    m_delay = args.media_delay
    request_max_count = args.request_max_count
    secrets_path = args.secrets_path
    secret_type = args.secrets_type
    endpoint_url = args.endpoint_url
    platform = args.platform

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(media_output_path, exist_ok=True)

    with open(secrets_path, "r") as f:
        secrets = json.load(f)[secret_type]

    start_date = datetime.strptime(start_date_str, query_time_format)
    end_date = datetime.strptime(end_date_str, query_time_format)

    request_delta = parse_timedelta(request_time_delta_str)

    all_dates = list(datetime_range(start_date, end_date, request_delta))
    for q_date_start, q_date_end in tqdm(all_dates, total=len(all_dates)):
        start_time = q_date_start.strftime(query_time_format)
        end_time = q_date_end.strftime(query_time_format)

        offset = 0
        while True:
            request_name = (
                q_date_start.strftime(file_time_format)
                + "-"
                + q_date_end.strftime(file_time_format)
                + "-"
                + f"{offset}"
            )
            completed_path = os.path.join(output_path, f"{request_name}.lock")
            if os.path.exists(completed_path):
                with open(completed_path) as f:
                    c_req = json.load(f)
                    num_results = c_req["num_results"]
            else:
                result_path = os.path.join(output_path, f"{request_name}.json")
                parameters = {
                    "token": secrets["token"],
                    "startDate": start_time,
                    "endDate": end_time,
                    "sortBy": "date",
                    "searchTerm": search_query,
                    "platforms": platform,
                    "language": language,
                    "count": request_max_count,
                    "offset": offset,
                }
                endpoint = format_parameters(endpoint_url, parameters)
                try:
                    response = requests.get(endpoint).json()
                    response_time = time.time()
                except Exception as e:
                    print(e)
                    time.sleep(10 * q_delay)
                    continue
                status = response["status"]
                if status != 200:
                    print(response)
                    time.sleep(10 * q_delay)
                    continue
                # ['posts', 'pagination', 'hitCount']
                results = response["result"]
                posts = results["posts"]
                num_results = len(posts)
                # 1 second delay between each post + random delay from 0 to 1
                media_delay = m_delay + random_delay(scale=0.25)
                posts = download_media(posts, media_output_path, media_delay, platform)
                with open(result_path, "w") as f:
                    json.dump(posts, f)
                with open(completed_path, "w") as f:
                    json.dump({"num_results": num_results}, f)
                process_time = time.time()
                process_delay = process_time - response_time
                sleep_delay = q_delay - process_delay
                if sleep_delay > 0.0:
                    time.sleep(sleep_delay)

            # if our request returned less than request_max_count then
            # no more offsets to check
            if num_results < request_max_count:
                break
            offset += request_max_count


if __name__ == "__main__":
    main()
