import json
import os
import time
from datetime import date, timedelta

import requests
from tqdm import tqdm


def format_endpoint(endpoint_url, fields):
    endpoint = endpoint_url + "?" + "&".join(fields)
    return endpoint


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n), start_date + timedelta(n + 1)


def main():
    # 2021-07-23 2021-07-24
    # start_date = date(2021, 7, 1)
    # v7 2021-10-19
    # start_date = date(2021, 7, 23)
    # start_date = date(2021, 10, 20)
    # 2020-12-18
    start_date = date(2019, 12, 18)
    end_date = date(2020, 12, 18)
    # wait 5 seconds between queries
    q_delay = 5

    # output_path = "/users/max/data/corpora/covid19-vaccine-twitter/raw-v7"
    output_path = "/users/max/data/corpora/covid19-vaccine-twitter/raw-v0"
    secrets_path = "private/secrets.json"
    with open(secrets_path, "r") as f:
        secrets = json.load(f)["twitter"]
    endpoint_url = "https://api.twitter.com/2/tweets/search/all"
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
    default_search_query = f"({covid_query}) ({vaccine_query})"

    query = f"query={default_search_query} lang:en -is:retweet"
    max_results = "max_results=500"
    tweet_fields = (
        "tweet.fields="
        "id,text,author_id,created_at,conversation_id,public_metrics,"
        "in_reply_to_user_id,referenced_tweets,lang,attachments,geo,"
        "possibly_sensitive,reply_settings,source,withheld,entities"
    )
    user_fields = (
        "user.fields=id,created_at,description,name,public_metrics,url," "username,verified,location,withheld,entities"
    )
    expansions = (
        "expansions=author_id,in_reply_to_user_id,referenced_tweets.id,"
        "referenced_tweets.id.author_id,attachments.media_keys,geo.place_id,"
        "entities.mentions.username"
    )
    media_fields = "media.fields=duration_ms,height,media_key,preview_image_url,type," "url,width,public_metrics"
    place_fields = "place.fields=contained_within,country,country_code,full_name,geo,id," "name,place_type"

    all_dates = list(date_range(start_date, end_date))
    for q_date_start, q_date_end in tqdm(all_dates, total=len(all_dates)):
        start_time = q_date_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time = q_date_end.strftime("%Y-%m-%dT%H:%M:%SZ")
        next_token = ""
        page_idx = 0
        completed_file = q_date_start.strftime("%Y%m%dT%H%M%SZ") + q_date_end.strftime("%Y%m%dT%H%M%SZ") + ".lock"
        completed_path = os.path.join(output_path, completed_file)
        if os.path.exists(completed_path):
            continue
        num_results = 0
        while next_token is not None:
            result_name = (
                q_date_start.strftime("%Y%m%dT%H%M%SZ") + q_date_end.strftime("%Y%m%dT%H%M%SZ") + f"-{page_idx}.json"
            )
            result_path = os.path.join(output_path, result_name)
            fields = [
                query,
                f"start_time={start_time}",
                f"end_time={end_time}",
                max_results,
                tweet_fields,
                user_fields,
                expansions,
                media_fields,
                place_fields,
            ]
            if len(next_token) > 0:
                fields.append(f"next_token={next_token}")
            endpoint = format_endpoint(endpoint_url, fields)
            headers = {"Authorization": f"Bearer {secrets['bearer_token']}"}
            try:
                results = requests.get(endpoint, headers=headers).json()
                with open(result_path, "w") as f:
                    json.dump(results, f)
            except Exception as e:
                print(e)
                time.sleep(10 * q_delay)
                continue
            if "meta" not in results:
                print(results)
                time.sleep(10 * q_delay)
                continue
            next_token = None
            if "next_token" in results["meta"]:
                next_token = results["meta"]["next_token"]
            result_count = results["meta"]["result_count"]
            time.sleep(q_delay)
            page_idx += 1
            num_results += result_count

        with open(completed_path, "w") as f:
            json.dump({"num_results": num_results, "num_pages": page_idx}, f)


if __name__ == "__main__":
    main()
