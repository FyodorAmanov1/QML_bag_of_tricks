"""Use a trusted saved estimator or accepted offline student on a new numeric CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np


def predict_csv(artifact_path,csv_path,output,features=None,id_column=None,state_budget=None):
    output = Path(output)
    if output.suffix.lower() != '.csv':
        raise ValueError('Prediction output must use a .csv suffix')
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Choose new prediction paths; previous outputs are preserved')
    artifact_path,csv_path = Path(artifact_path),Path(csv_path)
    artifact = joblib.load(artifact_path)  # trusted Python artifact only
    if isinstance(artifact,dict):
        model = artifact['model']
        names = artifact['features']
        if features is not None and features != names:
            raise ValueError('Features must match the fitted artifact order')
    else:
        model,names = artifact,features
        if not isinstance(names,list) or len(names)!=getattr(model,'n_features_in_',None):
            raise ValueError('Supply the original ordered feature names for a standalone student')
        if hasattr(model,'accepted_') and not model.accepted_:
            raise ValueError('Unaccepted student cannot be used for deployment prediction')
    with csv_path.open(encoding='utf-8-sig',newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames) or not set(names)<=set(reader.fieldnames) or (id_column and id_column not in reader.fieldnames):
            raise ValueError('CSV must contain unique headers and all fitted feature/ID columns')
        rows = list(reader)
    if not rows or len(rows)>20000 or any(None in row or None in row.values() for row in rows):
        raise ValueError('Require 1..20000 well-formed prediction records; batch larger files explicitly')
    x = np.array([[float(row[name]) if row[name].strip() else np.nan for name in names] for row in rows])
    before = dict(getattr(model,'resource_',{}))
    if state_budget is not None:
        if type(state_budget) is not int or state_budget<1 or not hasattr(model,'max_state_preparations'):
            raise ValueError('Positive state-budget override applies only to native quantum predictors')
        model.max_state_preparations = before.get('state_preparations',0)+state_budget
    predictions = model.predict(x)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['record_id','prediction'])
        writer.writerows((row[id_column] if id_column else i,prediction) for i,(row,prediction) in enumerate(zip(rows,predictions)))
    after = getattr(model,'resource_',{})
    manifest = dict(records=len(rows),features=names,artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    input_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                    simulated_native_preparations=after.get('state_preparations',0)-before.get('state_preparations',0),
                    physical_qpu_jobs=after.get('qpu_jobs',0)-before.get('qpu_jobs',0),
                    output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--features',nargs='+')
    parser.add_argument('--id-column')
    parser.add_argument('--state-budget',type=int)
    args = parser.parse_args(argv)
    print(json.dumps(predict_csv(args.model,args.csv,args.output,args.features,args.id_column,args.state_budget),indent=2))


if __name__ == '__main__':
    main()
