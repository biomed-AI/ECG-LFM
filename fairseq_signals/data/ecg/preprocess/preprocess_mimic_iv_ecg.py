"""
Data pre-processing:
    1. filter samples to have at least 2 corresponding sessions according to `patient_id`
    2. encode labels (patient id) and random crop data
"""

import argparse
import os
import pandas as pd
import wfdb
import scipy.io
import numpy as np
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="root directory containing data files to pre-process"
    )
    parser.add_argument(
        "--dest", type=str, metavar="DIR",
        help="output directory"
    )
    parser.add_argument(
        "--leads",
        default="0,1,2,3,4,5,6,7,8,9,10,11",
        type=str,
        help="comma separated list of lead numbers. (e.g. 0,1 loads only lead I and lead II)"
        "note that the order is following: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]"
    )
    parser.add_argument(
        "--sample-rate",
        default=500,
        type=int,
        help="if set, data must be sampled by this sampling rate to be processed"
    )
    parser.add_argument(
        "--sec", default=5, type=int,
        help="seconds to randomly crop to"
    )
    parser.add_argument(
        "--only-norm",
        default=False,
        type=bool,
        help="whether to preprocess only normal samples (normal sinus rhythms)"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")

    return parser

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    leads = args.leads.replace(' ','').split(',')
    leads_to_load = [int(lead) for lead in leads]

    #csv = pd.read_csv(os.path.join(dir_path, 'ptbxl_database.csv'))

    database = pd.read_csv(os.path.join(dir_path, "machine_measurements.csv")).set_index("study_id")
    record_list = pd.read_csv(os.path.join(dir_path, "record_list.csv"))

    study_ids = record_list["study_id"].to_numpy()
    patient_ids = record_list['subject_id'].to_numpy()
    fnames = record_list['path'].to_numpy()

    reports = []
    n_reports = 18
    # for study_id in tqdm(study_ids):
    #     report_txt = ""
    #     for j in range(n_reports):
    #         report = database.loc[study_id][f"report_{j}"]
    #         if type(report) == str:
    #             report_txt += report + " "
    #     report_txt = report_txt[:-1]
    #     reports.append(report_txt)

    table = dict()
    for fname, patient_id, scp_code in tqdm(zip(fnames, patient_ids, fnames)):
        if args.only_norm and 'NORM' in eval(scp_code):
            if patient_id in table:
                table[patient_id] += ',' + os.path.join(dir_path, fname)
            else:
                table[patient_id] = os.path.join(dir_path, fname)
        else:
            if patient_id in table:
                table[patient_id] += ',' + os.path.join(dir_path, fname)
            else:
                table[patient_id] = os.path.join(dir_path, fname)

    filtered = {k: v for k, v in table.items() if len(v.split(',')) >= 2}

    np.random.seed(args.seed)
    print(len(table.items()))
    print(list(filtered.items())[0])

    n=0
    for pid, fnames in tqdm(filtered.items()):
        for fname in fnames.split(','):
            basename = os.path.basename(fname)
            record = wfdb.rdsamp(fname)
            
            sample_rate = record[1]['fs']
            record = record[0].T

            if args.sample_rate and sample_rate != args.sample_rate:
                continue
            
            if np.isnan(record).any():
                print(f"detected nan value at: {fname}, so skipped")
                continue
            
            length = record.shape[-1]
            pid = int(pid)

            start = np.random.randint(length - (args.sec * sample_rate))

            data = {}
            data['patient_id'] = pid
            data['curr_sample_rate'] = sample_rate
            data['feats'] = record[leads_to_load, start: start + (args.sec * sample_rate)]
            scipy.io.savemat(os.path.join(dest_path, f"{pid}_{basename}.mat"), data)
            n+=1


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)