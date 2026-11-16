"""Strict numeric CSV contract and disjoint IID/group/chronological splits."""
import csv
from itertools import islice
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def load_data(config, base=Path('.')):
    data = config['data']
    source = Path(data['path'])
    if not source.is_absolute():
        source = Path(base)/source
    ceiling = 20000 if config.get('adapter') == 'batched_features' else 512
    limit = config.get('budget',{}).get('max_rows',ceiling)
    byte_limit = config.get('budget',{}).get('max_csv_bytes',32*1024*1024)
    if type(limit) is not int or not 20 <= limit <= ceiling:
        raise ValueError(f'max_rows must be an integer in 20..{ceiling}')
    if type(byte_limit) is not int or byte_limit<1 or source.stat().st_size>byte_limit:
        raise ValueError('CSV exceeds the declared max_csv_bytes file-size budget')
    payload = source.read_bytes()
    with source.open(encoding='utf-8-sig',newline='') as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames
        if not names or len(set(names)) != len(names):
            raise ValueError('Require a CSV header with unique column names')
        rows = list(islice(reader,limit+1))
    if not 20 <= len(rows) <= limit:
        raise ValueError(f'Require 20..max_rows records, max_rows <= {ceiling}; no silent subsampling')
    features, target = data['features'],data['target']
    split = data['split']
    excluded = {target,split.get('group_column'),split.get('time_column'),data.get('id_column')}
    if not isinstance(features,list) or not 1 <= len(features) <= 64 or len(set(features)) != len(features) or set(features)&excluded or not set(features).issubset(names) or target not in names:
        raise ValueError('List 1..64 unique numeric feature columns; exclude target, ID, group and time')
    if any(None in row or None in row.values() for row in rows):
        raise ValueError('Malformed CSV row width')
    x = np.array([[float(row[name]) if row[name].strip() else np.nan for name in features] for row in rows])
    if np.isinf(x).any() or any(not row[target].strip() for row in rows):
        raise ValueError('Reject infinite features and empty targets')
    task = config['task']
    y = np.array([row[target] for row in rows],dtype=str if task == 'classification' else float)
    if task == 'regression' and not np.isfinite(y).all():
        raise ValueError('Require finite regression targets')
    id_column = data.get('id_column')
    if id_column and id_column not in names:
        raise ValueError('Missing ID column')
    ids = [row[id_column] for row in rows] if id_column else list(range(len(rows)))
    if len(set(ids)) != len(ids):
        raise ValueError('IDs must be unique; repeated entities belong in the group column')
    for name in (split.get('group_column'),split.get('time_column')):
        if name and (name not in names or any(not row[name].strip() for row in rows)):
            raise ValueError('Missing group/time values')
    return x,y,rows,ids,payload


def split_indices(y, rows, specification, seed, task, ids=None):
    indices = np.arange(len(y))
    kind = specification['kind']
    if kind == 'iid':
        train,other = train_test_split(indices,test_size=.4,random_state=seed,
                                       stratify=y if task == 'classification' else None)
        validation,test = train_test_split(other,test_size=.5,random_state=seed+1,
                                           stratify=y[other] if task == 'classification' else None)
    elif kind == 'group':
        groups = np.array([row[specification['group_column']] for row in rows])
        if len(set(groups)) < 5:
            raise ValueError('Need at least five independent groups for a three-way split')
        train,other = next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=seed).split(indices,groups=groups))
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1).split(other,groups=groups[other]))
        validation,test = other[a],other[b]
    elif kind == 'chronological':
        # Strict ISO dates/times only, with unique timestamps. Panel/repeated
        # times require an explicit grouped temporal adapter, never a row sort.
        from datetime import datetime
        times = [datetime.fromisoformat(row[specification['time_column']]) for row in rows]
        if len(set(times)) != len(times):
            raise ValueError('Repeated timestamps need a grouped temporal adapter')
        order = np.array(sorted(indices,key=lambda i:times[i]))
        gap = specification.get('gap_rows',0)
        if type(gap) is not int or gap < 0:
            raise ValueError('gap_rows must be a nonnegative integer')
        a,b = int(.6*len(order)),int(.8*len(order))
        train,validation,test = order[:a],order[a+gap:b],order[b+gap:]
    elif kind == 'predefined':
        ids = list(range(len(y))) if ids is None else ids
        positions = {identity:index for index,identity in enumerate(ids)}
        selected = [specification.get(key+'_ids',[]) for key in ('train','validation','test')]
        if any(not isinstance(values,list) for values in selected):
            raise ValueError('Predefined partitions must be lists of original row IDs')
        flat = [identity for values in selected for identity in values]
        if len(set(flat)) != len(flat) or not set(flat) <= set(positions):
            raise ValueError('Predefined split IDs overlap, repeat or are absent')
        train,validation,test = [np.array([positions[identity] for identity in values],dtype=int) for values in selected]
    else:
        raise ValueError('split.kind must be iid, group, chronological or predefined')
    if min(map(len,(train,validation,test))) < 4:
        raise ValueError('Each split requires at least four records')
    if task == 'classification' and (len(set(y[train])) < 2 or any(set(y[a])-set(y[train]) for a in (validation,test))):
        raise ValueError('Training needs all evaluation classes; do not break groups/time to repair this')
    return dict(train=train,validation=validation,test=test)
