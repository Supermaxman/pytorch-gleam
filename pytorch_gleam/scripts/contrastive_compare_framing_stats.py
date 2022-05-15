import argparse
import pandas as pd
import ujson as json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-f", "--frame_path", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    frame_path = args.frame_path

    print("Loading data...")
    with open(frame_path) as f:
        frames = json.load(f)
    num_total_known_framings = len(frames)
    known_framing_ids = set(frames.keys())

    df = pd.read_excel(input_path, index_col=0)
    is_same = df['Same']
    is_framing = df['Framing']

    num_true_found_framings = df['Framing'].sum()
    num_total_found_framings = len(df)
    percent_true_found_framings = num_true_found_framings / num_total_found_framings
    print(f'True discovered framings: {num_true_found_framings}/{num_total_found_framings} ({100*percent_true_found_framings:.0f}%)')

    found_known_framings = set()
    for idx, row in df.iterrows():
        if row['Same'] == 1:
            found_known_framings.add(row['closest_framing'])

    num_found_known_framings = len(found_known_framings)
    percent_found_known_framings = num_found_known_framings / num_total_known_framings
    print(f'Known framings re-discovered: {num_found_known_framings}/{num_total_known_framings} ({100*percent_found_known_framings:.0f}%)')



if __name__ == "__main__":
    main()
