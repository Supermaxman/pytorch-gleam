import argparse
import json
import os
import subprocess
import time
from typing import Dict

import wandb
import yaml
from openai import OpenAI
from termcolor import colored


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


def prune_config(config: Dict[str, str], skip_config_keys: set):
    for k, v in list(config.items()):
        if k in skip_config_keys:
            del config[k]
        elif k == "init_args":
            for k2, v2 in list(v.items()):
                if k2 in skip_config_keys:
                    del v[k2]
                    continue
                elif isinstance(v2, dict):
                    prune_config(v2, skip_config_keys)
                config[k2] = v2
            del config[k]
        elif isinstance(v, dict):
            prune_config(v, skip_config_keys)
    return config


# recursively update the config with the hyperparameters if they match any keys in the config
def update_config(config: Dict[str, str], hyperparameters: Dict[str, str], ex_name: str, logs_path=None, project=None):
    for k, v in config.items():
        if k in hyperparameters:
            config[k] = hyperparameters[k]
        elif k == "logger":
            logs_path = v["init_args"]["save_dir"]
            project = v["init_args"]["project"]
        elif k == "default_root_dir":
            path, _ = os.path.split(v)
            config[k] = os.path.join(path, ex_name)
        elif isinstance(v, dict):
            config[k], logs_path, project = update_config(config[k], hyperparameters, ex_name, logs_path, project)
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


def print_message(message, fo):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    if message["role"] == "system":
        print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        fo.write(f"system: {message['content']}\n\n")
    elif message["role"] == "user":
        print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        fo.write(f"user: {message['content']}\n\n")
    elif message["role"] == "assistant" and message.get("tool_calls"):
        tool_calls = message["tool_calls"]
        lines = ["assistant:"]
        for tool_call in tool_calls:
            lines.append(f"  tool ({tool_call.function.name}):")
            try:
                hyperparameters = json.loads(tool_call.function.arguments)
                for k, v in hyperparameters.items():
                    lines.append(f"    {k}: {v}")
            except json.decoder.JSONDecodeError:
                hyperparameters = tool_call.function.arguments
                lines.append(f"    {hyperparameters}")
        print(colored("\n".join(lines) + "\n", role_to_color[message["role"]]))
        fo.write("\n".join(lines) + "\n\n")
    elif message["role"] == "assistant":
        print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        fo.write(f"assistant: {message['content']}\n\n")
    elif message["role"] == "tool":
        print(colored(f"tool ({message['name']}): {message['content']}", role_to_color[message["role"]]))
        print(colored(f"  {message['experiment']}\n", role_to_color[message["role"]]))
        fo.write(f"tool ({message['name']}): {message['content']}\n")
        fo.write(f"  {message['experiment']}\n\n")


def run(hyperparameters: Dict[str, str], config_path: str, i: int, org: str):
    ex_config_path, logs_path, project = create_new_config(hyperparameters, config_path, i)
    print(f"Running experiment: {ex_config_path}\n")
    try:
        start = time.time()
        # TODO eventually use stdout out = ...
        subprocess.run(
            ["python", "pytorch_gleam/ex/gleam.py", "fit", "--config", f"{ex_config_path}"],
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
        # TODO could fail
        api = wandb.Api()
        # TODO could fail
        run = api.run(f"{org}/{project}/{ex_id}")
        # TODO could fail
        summary: dict = run.summary
        # TODO consider entire run history, not just the last values
        # https://docs.wandb.ai/guides/track/public-api-guide#runhistory
        outputs = []
        for k, v in summary.items():
            if k.startswith("_") or isinstance(v, dict):
                continue
            outputs.append(f"{k}: {v}")
        outputs.append(f"seconds: {seconds}")
        outputs = "\n".join(outputs)

    except subprocess.CalledProcessError as e:
        # TODO possibly include stdout
        # TODO handle errors better
        outputs = e.stderr.decode()
    return outputs, ex_config_path


def main():
    parser = argparse.ArgumentParser()
    # gpt-4-1106-preview or gpt-3.5-turbo-1106
    parser.add_argument(
        "--model", type=str, default="gpt-4-1106-preview", choices=["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Fine-tuning a BERT-base model for stance detection between tweets and frames of communication.",
        help="Description of the experiment to optimize.",
    )
    parser.add_argument("--experiments", type=int, default=10, help="Number of experiments to run.")
    parser.add_argument(
        "--hyperparameters",
        nargs="+",
        default=[
            "learning_rate:number",
            "batch_size:integer",
            "accumulate_grad_batches:integer",
            "lr_warm_up:number",
            "max_epochs:integer",
            "weight_decay:number",
        ],
        help="List of hyperparameters to optimize.",
    )
    parser.add_argument(
        "--skip_config_keys",
        nargs="+",
        default=[
            "class_path",
            "seed_everything",
            "threshold",
            "check_val_every_n_epoch",
            "deterministic",
            "num_sanity_val_steps",
            "accelerator",
            "devices",
            "default_root_dir",
            "enable_checkpointing",
            "logger",
            "callbacks",
            "label_name",
            "num_workers",
            "frame_path",
            "train_path",
            "val_path",
            "test_path",
            "predict_path",
        ],
        help="List of hyperparameters to optimize.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--org", type=str, default="hltri", help="Organization to use for wandb.")
    parser.add_argument("--metric", type=str, default="val_f1", help="Metric to optimize.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--delay", type=int, default=10, help="Minimum delay between experiments in seconds.")
    parser.add_argument("--output", type=str, default="output.txt", help="Output file for conversation.")
    parser.add_argument("--train_size", type=int, default=10250, help="Size of the training set.")
    parser.add_argument("--val_size", type=int, default=1115, help="Size of the validation set.")
    parser.add_argument(
        "--device",
        type=str,
        default="NVIDIA TITAN V w/ 12GB VRAM",
        help="Accelerator device to help inform hyperparameter selection.",
    )

    args = parser.parse_args()

    model = args.model
    description = args.description
    experiments = args.experiments
    config_path = args.config
    seed = args.seed
    hyperparameter_names_types = [h.split(":") for h in args.hyperparameters]
    hyperparameter_names = [h for (h, _) in hyperparameter_names_types]
    skip_config_keys = set(args.skip_config_keys)
    org = args.org
    metric = args.metric
    direction = args.direction
    delay = args.delay
    output = args.output
    device = args.device
    train_size = args.train_size
    val_size = args.val_size

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = prune_config(config, skip_config_keys)
    config_str = yaml.dump(config)

    client = OpenAI()

    messages = []
    system_prompts = [
        "You are an assistant to a graduate research scientist.",
        "You will assist in optimizing the hyperparameters of ongoing experiments.",
        "You will be given a description of the experiment and a set of hyperparameters to optimize.",
        "Results of each experiment will be provided to you.",
        "You will continue to propose new hyperparameters and run experiments with the goal of improving the results.",
        "Furthermore, after each experiment, you will be asked to discuss the results and what was learned.",
        "Only propose one set of hyperparameters for a single experiment.",
    ]
    system_prompt = " ".join(system_prompts)

    messages.append({"role": "system", "content": system_prompt})

    user_prompts = [
        f"This experiment involves: {description}",
        f"The training set will contain {train_size:,} examples, while the validation set will contain {val_size:,} examples.",
        f"Experiments will be performed on a {device}.",
        "The initial configuration file for this experiment is:",
        f"```yaml\n{config_str}```",
        f"The hyperparameters to optimize are: {', '.join(hyperparameter_names)}",
        "Please begin to propose hyperparameters.",
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
                            "type": t,
                            "description": f"The value for the {h} hyperparameter.",
                        }
                        for (h, t) in hyperparameter_names_types
                    },
                    "required": hyperparameter_names,
                },
            },
        },
    ]

    with open(output, "w") as fo:
        start_idx = get_ex_idx(config_path)
        for message in messages:
            print_message(message, fo)

        for i in range(start_idx + 1, start_idx + experiments + 1):
            start = time.time()
            # TODO add @retry
            # TODO could fail
            chat_completion = client.chat.completions.create(
                messages=messages, model=model, max_tokens=512, seed=seed, top_p=0.7, tool_choice="run", tools=tools
            )
            choice = chat_completion.choices[0]
            # TODO could fail
            if choice.finish_reason != "tool_calls":
                raise ValueError(f"Invalid finish reason: {choice.finish_reason}")
            message = choice.message
            tool_message = {
                "role": "assistant",
                "tool_calls": message.tool_calls,
                "content": "",  # hack to get the tool calls to show up
            }
            messages.append(tool_message)
            print_message(tool_message, fo)
            for tool_call in message.tool_calls:
                if tool_call.function.name == "run":
                    tool_call_id = tool_call.id
                    # TODO could fail
                    try:
                        hyperparameters = json.loads(tool_call.function.arguments)
                    except json.decoder.JSONDecodeError:
                        raise ValueError(f"Invalid hyperparameter JSON: {tool_call.function.arguments}")
                    results, ex_config_path = run(hyperparameters, config_path, i, org)
                    response_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_call.function.name,
                        "content": results,
                    }
                    messages.append(response_message)
                    print_message(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_call.function.name,
                            "content": results,
                            "experiment": ex_config_path,
                        },
                        fo,
                    )

            end = time.time()
            seconds = end - start
            if delay > seconds:
                time.sleep(delay - seconds)

            start = time.time()
            # TODO add @retry
            # TODO could fail
            # also have the model discuss the results and what was learned
            chat_completion = client.chat.completions.create(
                messages=messages, model=model, max_tokens=512, seed=seed, top_p=0.7, tool_choice="none", tools=tools
            )
            choice = chat_completion.choices[0]
            message = choice.message
            summary_message = {
                "role": "assistant",
                "content": message.content,
            }
            messages.append(summary_message)
            print_message(summary_message, fo)

            # https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
            # may not be necessary
            # continue_message = {"role": "user", "content": "Continue"}
            # messages.append(continue_message)
            # print_message(continue_message, fo)
            end = time.time()
            seconds = end - start
            if delay > seconds:
                time.sleep(delay - seconds)


if __name__ == "__main__":
    main()
