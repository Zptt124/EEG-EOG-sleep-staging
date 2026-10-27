"""Rebuild W2 budgets from B's existing ranking; never shuffle or read val/test NPZ.

Default uses checked-in training-record counts (stdlib only). --processed reads
only the labels member of training NPZ files and requires NumPy. Existing output
files are refused. This prepares reviewable manifests, not training results.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('W', 'N1', 'N2', 'N3', 'REM')
BUDGETS = ((10, 5), (25, 14), (50, 27), (100, 54))
SUBSET_FIELDS = ('sampling_seed', 'subject_id', 'record_ids', 'n_records',
                 'n_epochs', 'sampling_rank', 'label10', 'label25', 'label50', 'label100')
RECORD_FIELDS = ('subject_id', 'record_id', 'n_epochs', *STAGES)
COUNT_FIELDS = ('sampling_seed', 'budget', 'labeled_subjects', 'unlabeled_subjects',
                'subject_percent', 'labeled_records', 'labeled_epochs',
                'epoch_percent', *STAGES)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_sources(split_rows, v1_rows):
    subjects = [r['subject_id'] for r in split_rows]
    require(len(subjects) == len(set(subjects)), 'Duplicate subject in split')
    require(Counter(r['split'] for r in split_rows) == {'train': 54, 'val': 8, 'test': 16},
            'Expected fixed 54/8/16 split')
    train = {r['subject_id']: r for r in split_rows if r['split'] == 'train'}
    all_records = [rec for r in split_rows for rec in r['record_ids'].split(';')]
    require(all(all_records) and len(all_records) == len(set(all_records)),
            'Empty or repeated record across split')
    for row in split_rows:
        require(all(rec.startswith(row['subject_id']) for rec in row['record_ids'].split(';')),
                'Record does not belong to split subject')
    require(len(v1_rows) == 54 and {r['subject_id'] for r in v1_rows} == set(train),
            'v1 must contain exactly the 54 training subjects')
    require({int(r['sampling_seed']) for r in v1_rows} == {2024}, 'Expected primary seed2024')
    require(sorted(int(r['sampling_rank']) for r in v1_rows) == list(range(1, 55)),
            'Ranks must be unique 1..54')
    for row in v1_rows:
        records = row['record_ids'].split(';')
        require(records == train[row['subject_id']]['record_ids'].split(';'),
                'v1 record list differs from fixed split')
        require(int(row['n_records']) == len(records), 'v1 record count differs')
        require(int(row['n_epochs']) > 0, 'v1 epoch count must be positive')
        for budget, size in BUDGETS[1:]:
            require(int(row[f'label{budget}']) == int(int(row['sampling_rank']) <= size),
                    'Existing v1 budget flag does not match rank')
    return v1_rows


def validate_record_counts(v1_rows, counts):
    expected = {rec: row['subject_id'] for row in v1_rows for rec in row['record_ids'].split(';')}
    require(len(counts) == len(expected) and {r['record_id'] for r in counts} == set(expected),
            'Record counts must cover training records exactly once (no val/test)')
    indexed = {}
    for row in counts:
        rec = row['record_id']
        require(row['subject_id'] == expected[rec], 'Record subject mismatch')
        values = [int(row[s]) for s in STAGES]
        require(all(v >= 0 for v in values) and sum(values) == int(row['n_epochs']) > 0,
                'Invalid record class counts')
        indexed[rec] = row
    for row in v1_rows:
        require(sum(int(indexed[r]['n_epochs']) for r in row['record_ids'].split(';'))
                == int(row['n_epochs']), 'Training labels differ from v1 epoch counts')
    return indexed


def count_processed(v1_rows, processed):
    import numpy as np
    counts = []
    for row in v1_rows:
        for rec in row['record_ids'].split(';'):
            with np.load(Path(processed) / f'{rec}.npz', allow_pickle=False) as data:
                labels = data['labels']  # Signal arrays are not loaded.
            require(labels.ndim == 1 and labels.dtype.kind in 'iu', 'Labels must be integer vector')
            require(bool(np.all((labels >= 0) & (labels <= 4))), 'Invalid training label')
            values = np.bincount(labels.astype(np.int64), minlength=5)
            counts.append(dict(subject_id=row['subject_id'], record_id=rec, n_epochs=len(labels),
                               **dict(zip(STAGES, map(int, values)))))
    return counts


def build_tables(v1_rows, counts):
    indexed = validate_record_counts(v1_rows, counts)
    # Preserve v1 row order and every existing value; append only the new flag.
    subsets = [dict(row, label10=int(int(row['sampling_rank']) <= 5)) for row in v1_rows]
    total = sum(int(r['n_epochs']) for r in v1_rows)
    summaries = []
    for budget, size in BUDGETS:
        selected = [r for r in subsets if int(r[f'label{budget}']) == 1]
        require(len(selected) == size, 'Budget size mismatch')
        records = [rec for row in selected for rec in row['record_ids'].split(';')]
        classes = {s: sum(int(indexed[r][s]) for r in records) for s in STAGES}
        n_epochs = sum(classes.values())
        summaries.append(dict(sampling_seed=2024, budget=f'label{budget}', labeled_subjects=size,
            unlabeled_subjects=54-size, subject_percent=f'{100*size/54:.8f}',
            labeled_records=len(records), labeled_epochs=n_epochs,
            epoch_percent=f'{100*n_epochs/total:.8f}', **classes))
    return subsets, summaries, sorted(counts, key=lambda r: r['record_id'])


def write_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, default=ROOT / 'manifests/split_subjects.csv')
    parser.add_argument('--v1', type=Path, default=ROOT / 'manifests/label_subsets_v1.csv')
    parser.add_argument('--record-counts', type=Path, default=ROOT / 'manifests/label_record_counts_v2.csv')
    parser.add_argument('--processed', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows = validate_sources(read_csv(args.split), read_csv(args.v1))
    counts = count_processed(rows, args.processed) if args.processed else read_csv(args.record_counts)
    subsets, summaries, records = build_tables(rows, counts)
    outputs = [('label_subsets_v2.csv', SUBSET_FIELDS, subsets),
               ('label_counts_v2.csv', COUNT_FIELDS, summaries),
               ('label_record_counts_v2.csv', RECORD_FIELDS, records)]
    for filename, _, _ in outputs:
        if (args.output_dir / filename).exists():
            raise FileExistsError(f'Refusing to overwrite {args.output_dir / filename}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, fields, table in outputs:
        write_csv(args.output_dir / filename, fields, table)
    for row in summaries:
        print(f"{row['budget']}: {row['labeled_subjects']} subjects, "
              f"{row['labeled_records']} records, {row['labeled_epochs']} epochs, N1={row['N1']}")


if __name__ == '__main__':
    main()
