import argparse
import json
import os
import subprocess
import time
from typing import Dict

import wandb
import yaml
from openai import OpenAI


def get_ex_idx(config_path: str):
    _, file_name = os.path.split(config_path)
    ex_name = file_name.split(".")[0]
    try:
        if "-" not in ex_name:
            return 0
        # v1, v2, v3, etc.
        version = ex_name.split("-")[-1]
        ex_idx = int(version[1:])
    except ValueError:
        ex_idx = 0
    return ex_idx


def get_new_ex_path(config_path: str, i: int):
    ex_path, file_name = os.path.split(config_path)
    ex_name = file_name.split(".")[0]
    # bert, bert-v1, bert-v2, etc.
    if "-" not in ex_name:
        new_ex_name = f"{ex_name}-v{i}.yaml"
    else:
        version = ex_name.split("-")[-1]
        new_ex_name = f"{ex_name[:-len(version)]}v{i}.yaml"
    new_config_path = os.path.join(ex_path, new_ex_name)
    return new_config_path


# recursively update the config with the hyperparameters if they match any keys in the config
def update_config(config: Dict[str, str], hyperparameters: Dict[str, str], ex_name: str, logs_path=None, project=None):
    for k, v in config.items():
        if k in hyperparameters:
            config[k] = hyperparameters[k]
        elif isinstance(v, dict):
            config[k], logs_path, project = update_config(config[k], hyperparameters, logs_path, project)
        elif k == "default_root_dir":
            path, _ = os.path.split(v)
            config[k] = os.path.join(path, ex_name)
        elif k == "logger":
            logs_path = config[k]["init_args"]["save_dir"]
            project = config[k]["init_args"]["project"]
    return config, logs_path, project


def create_new_config(hyperparameters: Dict[str, str], config_path: str, i: int):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    new_ex_path = get_new_ex_path(config_path, i)
    new_ex_name = os.path.split(new_ex_path)[-1].split(".")[0]
    new_config, logs_path, project = update_config(config, hyperparameters, new_ex_name)

    with open(new_ex_path, "w") as f:
        yaml.dump(new_config, f)

    return new_ex_path, logs_path, project


def run(hyperparameters: Dict[str, str], config_path: str, i: int, org: str):
    ex_config_path, logs_path, project = create_new_config(hyperparameters, config_path, i)

    try:
        start = time.time()
        # TODO eventually use stdout out = ...
        subprocess.run(
            ["python", "pytorch_gleam/ex/gleam.py", "fit", "--config", f'"{ex_config_path}"'],
            capture_output=True,
            check=True,
        )
        end = time.time()
        seconds = end - start
        # outputs = out.stdout.decode()
        run_dir = os.path.join(logs_path, "wandb", "latest-run")
        ex_id = None
        for file in os.listdir(run_dir):
            if file.endswith(".wandb"):
                # run-ID.wandb
                ex_id = file.split(".")[0].split("-")[-1]
                break
        api = wandb.Api()
        run = api.run(f"{org}/{project}/{ex_id}")
        summary: dict = run.summary
        outputs = []
        for k, v in summary.items():
            if k.startswith("_") or isinstance(v, dict):
                continue
            outputs.append(f"{k}: {v}")
        outputs.append(f"seconds: {seconds}")
        outputs = "\n".join(outputs)

    except subprocess.CalledProcessError as e:
        # TODO possibly include stdout
        outputs = e.stderr.decode()
        # TODO handle errors better
        return outputs
    print(outputs)

    return outputs


def main():
    parser = argparse.ArgumentParser()
    # gpt-4-1106-preview or gpt-3.5-turbo-1106
    parser.add_argument(
        "--model", type=str, default="gpt-4-1106-preview", choices=["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Fine-tuning a BERT-base model for stance detection.",
        help="Description of the experiment to optimize.",
    )
    parser.add_argument("--experiments", type=int, default=10, help="Number of experiments to run.")
    parser.add_argument(
        "--hyperparameters", nargs="+", default=["learning_rate"], help="List of hyperparameters to optimize."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--org", type=str, default="hltri", help="Organization to use for wandb.")
    parser.add_argument("--metric", type=str, default="val_f1", help="Metric to optimize.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--delay", type=int, default=10, help="Minimum delay between experiments in seconds.")
    args = parser.parse_args()

    model = args.model
    description = args.description
    experiments = args.experiments
    config_path = args.config
    seed = args.seed
    hyperparameter_names = args.hyperparameters
    org = args.org
    metric = args.metric
    direction = args.direction
    delay = args.delay

    with open(config_path, "r") as f:
        config_str = f.read()

    client = OpenAI()

    messages = []
    system_prompts = [
        "You are an assistant to a graduate research scientist.",
        "You will assist in optimizing the hyperparameters of ongoing experiments.",
        "You will be given a description of the experiment and a set of hyperparameters to optimize.",
        "Results of each experiment will be provided to you.",
        "You will continue to propose new hyperparameters with the goal of improving the results.",
        "The goal is to find the best hyperparameters for the experiment in the shortest amount of time.",
    ]
    system_prompt = " ".join(system_prompts)
    print(system_prompt)

    messages.append({"role": "system", "content": system_prompt})

    user_prompts = [
        f"This experiment involves: {description}",
        "The initial configuration file for this experiment is:",
        f"```yaml\n{config_str}```",
        f"The hyperparameters to optimize are: {', '.join(hyperparameter_names)}",
        "Please begin to propose new hyperparameters to optimize the experiment.",
        "You can run the experiment with the given hyperparameters by using the `run` tool.",
        f"Please {direction} the {metric} metric.",
    ]
    messages.append({"role": "user", "content": "\n".join(user_prompts)})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "run",
                "description": "Run the experiment with the given hyperparameters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        h: {
                            "type": "string",
                            "description": f"the value for the {h} hyperparameter.",
                        }
                        for h in hyperparameter_names
                    },
                    "required": hyperparameter_names,
                },
            },
        },
    ]

    start_idx = get_ex_idx(config_path)

    for i in range(start_idx + 1, start_idx + experiments + 1):
        start = time.time()
        # TODO coild fail
        chat_completion = client.chat.completions.create(
            messages=messages, model=model, max_tokens=512, seed=seed, top_p=0.7, tool_choice="auto", tools=tools
        )
        choice = chat_completion.choices[0]
        # TODO could fail
        if choice.finish_reason == "tool_calls":
            message = choice.message
            messages.append(
                # {
                #     "role": "assistant",
                #     "tool_calls": message.tool_calls,
                #     "content": "",  # hack to get the tool calls to show up
                # }
                message
            )
            for tool_call in message.tool_calls:
                if tool_call.function.name == "run":
                    tool_call_id = tool_call.id
                    # TODO could fail
                    hyperparameters = json.load(tool_call.function.arguments)
                    results = run(hyperparameters, config_path, i, org)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_call.function.name,
                            "content": results,
                        }
                    )
        messages.append({"role": "user", "content": "Please propose new hyperparameters."})
        end = time.time()
        seconds = end - start
        if delay > seconds:
            time.sleep(delay - seconds)


if __name__ == "__main__":
    main()
