import os
import json
import requests
from pprint import pprint
from datetime import timedelta, date
import time
from tqdm import tqdm


def format_parameters(endpoint, parameters):
    p_text = "".join(
        [
            ("?" if idx == 0 else "&") + f"{key}={value}"
            for idx, (key, value) in enumerate(parameters.items())
        ]
    )
    return endpoint + p_text


def date_range(start_date, end_date):
    # inclusive range
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n), start_date + timedelta(n + 1)


if __name__ == "__main__":
    start_date = date(2020, 1, 1)
    end_date = date(2022, 1, 1)
    # 50 calls per minute * 1 minute / 60 seconds = 0.8333 calls per second
    q_delay = 50.0 / 60.0

    request_max_count = 100
    output_path = "/users/max/data/corpora/covid19-vaccine-facebook/raw-v1"
    secrets_path = "../private/secrets.json"
    with open(secrets_path, "r") as f:
        secrets = json.load(f)["twitter"]
    endpoint_url = "https://api.crowdtangle.com/posts/search"

    all_dates = list(date_range(start_date, end_date))
    for q_date_start, q_date_end in tqdm(all_dates, total=len(all_dates)):
        start_time = q_date_start.strftime("%Y-%m-%dT%H:%M:%S")
        end_time = q_date_end.strftime("%Y-%m-%dT%H:%M:%S")

        offset = 0
        while True:
            request_name = (
                q_date_start.strftime("%Y%m%dT%H%M%S")
                + "-"
                + q_date_end.strftime("%Y%m%dT%H%M%S")
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
                    "searchTerm": "(covid OR coronavirus OR coronavirus "
                    "OR corona OR covid-19 OR covid19 OR SARS-CoV-2 OR SARS OR SARS-CoV) "
                    "AND (vaccine OR vaccines OR vaccination OR vaccinations"
                    " OR vax OR vaxx OR vaxxed OR jab OR jabbed "
                    "OR vaccinate OR vaccinated OR vaccinates)",
                    "platforms": "facebook",
                    "language": "en",
                    "count": request_max_count,
                    "offset": offset,
                }
                endpoint = format_parameters(endpoint_url, parameters)
                try:
                    response = requests.get(endpoint).json()
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
                with open(result_path, "w") as f:
                    json.dump(posts, f)
                with open(completed_path, "w") as f:
                    json.dump({"num_results": num_results}, f)
                time.sleep(q_delay)

            # if our request returned less than request_max_count then
            # no more offsets to check
            if num_results < request_max_count:
                break
            offset += request_max_count
