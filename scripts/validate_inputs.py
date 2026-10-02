#!/usr/bin/env python3
"""Check the two local clinical inputs without fitting or printing participant rows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = {name: ROOT / 'upload' / name for name in [
        'Corrected_Clinical_Trial_Long_Data.csv', 'Appendix_A_Actual_Sequence.csv']}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit('Missing local research inputs: ' + ', '.join(missing)
                         + '. See upload/README.md. No data are downloaded.')
    data = pd.read_csv(paths['Corrected_Clinical_Trial_Long_Data.csv'])
    key = pd.read_csv(paths['Appendix_A_Actual_Sequence.csv'])
    required = {'ID', 'CaseSeries', 'Trial', 'RawTaskNumber', 'Accuracy',
                'direct_trust_item', 'decision_accept', 'Job_Category'}
    if not required.issubset(data.columns):
        raise ValueError('Clinical file lacks columns: ' + ', '.join(sorted(required - set(data.columns))))
    key_fields = ['CaseSeries', 'Trial', 'RawTaskNumber', 'Accuracy']
    if not set(key_fields).issubset(key.columns):
        raise ValueError('Sequence key lacks required columns: ' + ', '.join(key_fields))
    if len(data) != 1428 or data.ID.nunique() != 68:
        raise ValueError('This study-specific implementation expects 68 people and 1,428 rows.')
    if data[['ID', 'CaseSeries', 'Trial']].isna().any().any():
        raise ValueError('ID, CaseSeries, and Trial must be complete.')
    if data.duplicated(['ID', 'Trial']).any():
        raise ValueError('Duplicate participant-by-trial records.')
    if set(data.CaseSeries) != {'A', 'B'}:
        raise ValueError('Expected CaseSeries values A and B.')
    if not data.groupby('ID').CaseSeries.nunique().eq(1).all():
        raise ValueError('Each person must have one case series.')
    if not data.groupby('CaseSeries').ID.nunique().eq(34).all():
        raise ValueError('Expected 34 people per case series.')
    if not data.groupby('ID').Trial.apply(lambda x: sorted(x) == list(range(1, 22))).all():
        raise ValueError('Expected exactly trials 1 through 21 for each person.')
    ratings = data.direct_trust_item.to_numpy(float)
    if not (np.isfinite(ratings).all() and np.all(ratings == np.floor(ratings))
            and np.all((ratings >= 1) & (ratings <= 7))):
        raise ValueError('Ratings must be complete integers from 1 through 7.')
    if not data.decision_accept.dropna().isin([0, 1]).all():
        raise ValueError('Observed decisions must use 0/1; missing decisions remain missing.')
    if len(key) != 42 or key.duplicated(['CaseSeries', 'Trial']).any():
        raise ValueError('Expected 42 unique case-series/trial cells in sequence key.')
    joined = data.merge(key[key_fields], on=['CaseSeries', 'Trial'],
                        suffixes=('', '_key'), how='left', indicator=True, validate='many_to_one')
    if not joined['_merge'].eq('both').all():
        raise ValueError('Clinical rows are unmatched to the sequence key.')
    for field in ['RawTaskNumber', 'Accuracy']:
        if not joined[field].eq(joined[field + '_key']).all():
            raise ValueError('Sequence key mismatch: ' + field)
    if not data.Accuracy.isin(['Correct', 'Incorrect']).all():
        raise ValueError('Expected correctness labels Correct and Incorrect.')
    print(json.dumps({'status': 'pass', 'participants': 68, 'rows': 1428,
                      'case_series': 2, 'trials_per_person': 21,
                      'sequence_cells': 42, 'missing_choices': int(data.decision_accept.isna().sum())}, indent=2))


if __name__ == '__main__':
    main()
