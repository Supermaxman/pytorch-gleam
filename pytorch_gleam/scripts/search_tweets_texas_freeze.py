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


if __name__ == "__main__":
    # https://www.weather.gov/hgx/2021ValentineStorm
    # https://www.weather.gov/images/hgx/events/ValentinesStorm_2021/Timeline.png
    start_date = date(2021, 2, 11)
    end_date = date(2021, 2, 20)
    # wait 5 seconds between queries
    q_delay = 5.0

    output_path = "/users/max/data/corpora/texas-freeze/raw-v4"
    secrets_path = "private/secrets.json"
    with open(secrets_path, "r") as f:
        secrets = json.load(f)["twitter"]
    endpoint_url = "https://api.twitter.com/2/tweets/search/all"
    # consider hashtags like
    # #TexasFreeze, #TexasSnow and #TexasPowerOutages
    # v1-3
    # freeze_terms = []
    # v4
    # top 300 most common hashtags in texas geolocated tweets during freeze manually selected
    freeze_tags = [
        "#texasfreeze",
        "#texaswinterstorm2021",
        "#texasblackout",
        "#texaspoweroutage",
        "#texasweather",
        "#texaswinterstorm",
        "#snowmageddon2021",
        "#ercot",
        "#winterstorm2021",
        "#texasstrong",
        "#winterstorm",
        "#ercotfail",
        "#abbottfailedtexas",
        "#texaspowergrid",
        "#poweroutage",
        "#texaswinter",
        "#texassnow2021",
        "#texassnow",
        "#texasisclosed",
        "#snowpocalypse2021",
        "#icestorm2021",
        "#snowstorm2021",
        "#freezing",
        "#texasblizzard2021",
        "#staywarm",
        "#polarvortex",
        "#txblizzard",
        "#nopower",
        "#dallasweather",
        "#snowintexas",
        "#houstonfreeze",
        "#texaspoweroutages",
        "#power",
        "#staysafe",
        "#prayfortexas",
        "#rollingblackouts",
        "#snowday2021",
        "#austinweather",
        "#winterwonderland",
        "#winterweather",
        "#icestorm",
        "#texassnowday",
        "#texaslife",
        "#houstonpoweroutage",
        "#houstonweather",
        "#austinsnow",
        "#powergrid",
        "#articblast",
        "#electricity",
    ]
    tag_search_query = " OR ".join(freeze_tags)

    # texas + top 35 texas cities in pop size
    texas_terms = [
        "texas",
        "dallas",
        "houston",
        "austin",
        '"san antonio"',
        '"fort worth"',
        '"el paso"',
        "arlington",
        '"corpus christi"',
        "plano",
        "laredo",
        "lubbock",
        "irving",
        "garland",
        "amarillo",
        '"grand prairie"',
        "mckinney",
        "frisco",
        "brownsville",
        "killeen",
        "mesquite",
        "mcallen",
        "denton",
        "waco",
        "carrollton",
        "midland",
        '"round rock"',
        "abilene",
        "pearland",
        "richardson",
        "odessa",
        "beaumont",
        "lewisville",
        '"league city"',
        '"wichita falls"',
        '"san angelo"',
    ]
    texas_query = " OR ".join(texas_terms)
    # top 400 most common non-stopwords in texas geolocated tweets during freeze manually selected
    freeze_terms = [
        "freeze",
        "freezing",
        "froze",
        "frozen",
        "ice",
        "degrees",
        "degree",
        "power",
        "snow",
        "cold",
        "weather",
        "water",
        "warm",
        "fire",
        "winter",
        "rain",
        "storm",
        "outside",
        "electricity",
        "electric",
        "hot",
        "food",
        "❄",
        "🥶",
        "heat",
        "wind",
        "fire",
        "emergency",
        "blackout",
        "energy",
        "extreme",
        "extremes",
        "extremely",
    ]
    freeze_query = " OR ".join(freeze_terms)
    texas_search_query = f"({texas_query}) ({freeze_query})"

    # either hashtag match or texas state / city match and freeze term match
    default_search_query = f"{tag_search_query} OR ({texas_search_query})"
    # https://developer.twitter.com/en/use-cases/build-for-good/extreme-weather/texas-freeze
    # v1-3
    # search_terms = [
    #     # in v1
    #     # "has:geo",
    #     "lang:en",
    #     "-is:retweet",
    #     "-is:nullcast",
    #     # not in v1
    #     "-is:quote",
    #     # not in v2
    #     "-is:reply",
    #     # Texas place
    #     "place:texas",
    # ]
    # v4
    search_terms = [
        # in v1
        # "has:geo",
        "lang:en",
        "-is:retweet",
        "-is:nullcast",
        # not in v1
        "-is:quote",
        # not in v2
        "-is:reply",
        # v1-3 Texas place
        # "place:texas",
    ]
    # v1-3 false, v4 true
    use_query = True

    if use_query:
        search_terms.insert(0, default_search_query)

    query = "query=" + " ".join(search_terms)
    print(query)
    exit()
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
                response_time = time.time()
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
            process_time = time.time()
            process_delay = process_time - response_time
            sleep_delay = q_delay - process_delay
            if sleep_delay > 0.0:
                time.sleep(sleep_delay)
            page_idx += 1
            num_results += result_count

        with open(completed_path, "w") as f:
            json.dump({"num_results": num_results, "num_pages": page_idx}, f)
