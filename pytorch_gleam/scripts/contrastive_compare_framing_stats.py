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
    df = pd.read_excel(input_path, index_col=0)
    num_true_found_framings = df['Framing'].sum()
    num_total_found_framings = len(df)
    percent_true_found_framings = num_true_found_framings / num_total_found_framings
    print(f'Total judged discovered framings: {num_true_found_framings}/{num_total_found_framings} ({100*percent_true_found_framings:.1f}%)')

    found_known_framings = set()
    known_framings = set()
    found_unknown_framings = set()
    found_non_framings= set()
    for idx, row in df.iterrows():
        if row['Same'] == 1 and row['closest_framing'] not in known_framings:
            known_framings.add(row['closest_framing'])
            found_known_framings.add(idx)
        elif row['Framing'] == 1:
            found_unknown_framings.add(idx)
        else:
            found_non_framings.add(idx)
    num_found_known_framings = len(found_known_framings)
    percent_found_known_framings = num_found_known_framings / num_total_known_framings
    print(f'Known framings judged as re-discovered: {num_found_known_framings}/{num_total_known_framings} ({100*percent_found_known_framings:.1f}%)')

    num_found_unknown_framings = len(found_unknown_framings)
    num_found_non_framings = len(found_non_framings)
    percent_found_non_framings = num_found_non_framings / num_total_found_framings
    percent_found_known_framings_of_total_found = num_found_known_framings / num_total_found_framings
    percent_unknown_found_known_framings_of_total_found = num_found_unknown_framings / num_total_found_framings
    print(f'Unknown framings judged as discovered: {num_found_unknown_framings}')

    print(f'Total framings discovered distribution: ')
    print(f'  Known framings judged as re-discovered: {num_found_known_framings}/{num_total_found_framings} ({100*percent_found_known_framings_of_total_found:.1f}%)')
    print(f'  Unknown framings judged as discovered: {num_found_unknown_framings}/{num_total_found_framings} ({100*percent_unknown_found_known_framings_of_total_found:.1f}%)')
    print(f'  Non framings judged: {num_found_non_framings}/{num_total_found_framings} ({100*percent_found_non_framings:.1f}%)')

    tp = num_found_known_framings + num_found_unknown_framings
    fp = num_found_non_framings
    fn = num_total_known_framings - num_found_known_framings
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    k_recall = num_found_known_framings / num_total_known_framings

    f1 = (2.0 * precision * recall) / (precision + recall)

    print(f'Precision:  {100*precision:.1f}')
    print(f'Recall:     {100*recall:.1f}')
    print(f'F1-Score:   {100*f1:.1f}')

    results = {
        'F1': round(100*f1, 1),
        'Precision': round(100*precision, 1),
        'Recall': round(100*recall, 1),
        'Known-Recall': round(100*k_recall, 1),
    }

    c_txt = '\t'.join([k for k in results.keys()])
    v_txt = '\t'.join([f'{v}' for v in results.values()])
    print(c_txt)
    print(v_txt)


if __name__ == "__main__":
    main()
