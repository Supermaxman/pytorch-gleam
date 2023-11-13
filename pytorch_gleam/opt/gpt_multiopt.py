import argparse
import os

from pytorch_gleam.opt.gpt_opt import add_arguments, optimize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)

    parser.add_argument("--configs", nargs="+", help="Path to the experiment configuration files.")
    args = parser.parse_args()

    configs = args.configs

    assert len(configs) > 0, "Must provide at least one configuration file."

    for config in configs:
        assert os.path.exists(config), f"Config file {config} does not exist."

    for config in configs:
        print(f"Optimizing {config}...")
        print()
        text_output = config + ".txt"
        message_output = config + ".jsonl"
        optimize(
            model=args.model,
            description=args.description,
            experiments=args.experiments,
            config_path=config,
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
