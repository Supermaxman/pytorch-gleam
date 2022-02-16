import os
import json
import time

import requests
from pprint import pprint
from datetime import timedelta, date, datetime
from tqdm import tqdm


def format_parameters(endpoint, parameters):
    p_text = "".join(
        [
            ("?" if idx == 0 else "&") + f"{key}={value}"
            for idx, (key, value) in enumerate(parameters.items())
        ]
    )
    return endpoint + p_text


def datetime_range(start: datetime, end: datetime, step: timedelta):
    end -= step
    while start <= end:
        yield start, start + step
        start += step


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


if __name__ == "__main__":
    # start_date = date(2020, 1, 1)
    # end_date = date(2022, 1, 1)
    start_date_str = "2020-12-14T13:00:00"
    end_date_str = "2022-01-01T00:00:00"
    # hours, minutes, and seconds
    request_time_delta_str = "00:10:00"

    query_time_format = "%Y-%m-%dT%H:%M:%S"
    file_time_format = "%Y%m%dT%H%M%S"
    # time_delta_format = "%H:%M:%S"
    # 50 calls per minute * 1 minute / 60 seconds = 0.8333 calls per second
    q_delay = 50.0 / 60.0
    request_max_count = 100
    output_path = "/users/max/data/corpora/covid19-vaccine-facebook/raw-v2"
    secrets_path = "private/secrets.json"
    secret_type = "crowdtange"
    endpoint_url = "https://api.crowdtangle.com/posts/search"

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
                    "AND (vaccine OR vaccines OR vaccination OR vaccinations "
                    "OR vax OR vaxx OR vaxxed OR jab OR jabbed "
                    "OR vaccinate OR vaccinated OR vaccinates)",
                    "platforms": "facebook",
                    "language": "en",
                    "count": request_max_count,
                    "offset": offset,
                }
                endpoint = format_parameters(endpoint_url, parameters)
                try:
                    request_time = time.time()
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
                response_time = time.time()
                api_delay = response_time - request_time
                sleep_delay = q_delay - api_delay
                if sleep_delay > 0.0:
                    time.sleep(sleep_delay)

            # if our request returned less than request_max_count then
            # no more offsets to check
            if num_results < request_max_count:
                break
            offset += request_max_count
