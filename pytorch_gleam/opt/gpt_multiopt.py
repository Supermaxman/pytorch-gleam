import argparse
import os
import yaml

from pytorch_gleam.opt.gpt_opt import add_arguments, optimize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    parser.add_argument("--configs", nargs="+", help="Path to the experiment configuration files.")
    args = parser.parse_args()

    configs = args.configs

    assert len(configs) > 0, "Must provide at least one configuration file."

    print(f"Checking {len(configs):,} config files...")
    for config_path in configs:
        assert os.path.exists(config_path), f"Config file {config_path} does not exist."
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert "model" in config, f"Config file {config_path} does not specify a model."
        assert "data" in config, f"Config file {config_path} does not specify data."
        assert "trainer" in config, f"Config file {config_path} does not specify a trainer."
        print(f"  {config_path} OK.")

    print()
    print("All config files OK, proceeding to optimization.")
    print()
    for config_path in configs:
        print(f"Optimizing {config_path}...")
        print()
        text_output = config_path + ".txt"
        message_output = config_path + ".jsonl"
        optimize(
            model=args.model,
            description=args.description,
            experiments=args.experiments,
            config_path=config_path,
            seed=args.seed,
            hyperparameter_names_types=[h.split(":") for h in args.hyperparameters],
            skip_config_keys=set(args.skip_config_keys),
            org=args.org,
            metric=args.metric,
            metrics=args.metrics,
            direction=args.direction,
            delay=args.delay,
            text_output=text_output,
            message_output=message_output,
            device=args.device,
            train_size=args.train_size,
            val_size=args.val_size,
        )
