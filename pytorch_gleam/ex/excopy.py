import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    folder, exp_file_name = os.path.split(args.input)
    with open(args.input, "r") as f:
        lines = f.readlines()
    exp_name, exp_ext = os.path.splitext(exp_file_name)
    exp_version_idx = exp_name.rfind("v")
    exp_base_name = exp_name[: exp_version_idx + 1]
    exp_version = int(exp_name[exp_version_idx + 1 :])

    for i in range(1, args.k + 1):
        new_exp_version = exp_version + i
        new_exp_name = f"{exp_base_name}{new_exp_version}"
        output_path = os.path.join(folder, f"{new_exp_name}{exp_ext}")
        with open(output_path, "w") as f:
            for line in lines:
                line = line.replace(f"v{exp_version}", f"v{new_exp_version}")
                f.write(line)


if __name__ == "__main__":
    main()
