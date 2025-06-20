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
import ast
import linecache

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

def get_scpcode_result(code_list, report):
    dic = ast.literal_eval(report)
    for code in code_list:
        if code in list(dic.keys()):
            number = dic[code]
            if number>0:
                return True
    
    return False

def main(args):
    dir_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    leads = args.leads.replace(' ','').split(',')
    leads_to_load = [int(lead) for lead in leads]

    csv = pd.read_csv(os.path.join(dir_path, 'ptbxl_database.csv'))
    
    patient_ids = csv['patient_id'].to_numpy()
    fnames = csv['filename_hr'].to_numpy()
    scp_codes = csv['scp_codes']
    ages = csv['age']
    sexs = csv['sex']
    print(os.path.join(dir_path, 'ptbxl_database.csv'))
    print(csv)
    for fname, patient_id, scp_code, age, sex in tqdm(zip(fnames, patient_ids, scp_codes, ages, sexs)):
        fname = os.path.join(dir_path, fname)
        basename = int(os.path.basename(fname).split('_hr')[0])
        record = wfdb.rdsamp(fname)
        
        sample_rate = record[1]['fs']
        
        record = record[0].T
        
        if args.sample_rate and sample_rate != args.sample_rate:
                continue
            
        if np.isnan(record).any():
            print(f"detected nan value at: {fname}, so skipped")
            continue

        length = record.shape[-1]
        pid = int(patient_id)

        label = np.zeros(1, dtype=bool)

        for i in range(1):         
            if get_scpcode_result(['IMI', 'AMI', 'LMI', 'ILMI', 'ASMI', 'ALMI', 'PMI', 'IPLMI', 'IPMI'], scp_code): #MI
                label[i] = 1
        
        for i, seg in enumerate(range(0, length, int(args.sec * sample_rate))):
            
            data = {}
            data['age'] = age
            data['sex'] = sex
            data['label'] = label
            data['patient_id'] = pid
            data['curr_sample_rate'] = sample_rate
            if seg + args.sec * sample_rate <= length:
                data['feats'] = record[leads_to_load, seg: int(seg + args.sec * sample_rate)]
                #scipy.io.savemat(os.path.join(dest_path, os.path.basename(fname) + f"_{i}.mat"), data)
                scipy.io.savemat(os.path.join(dest_path, str(pid)+'_'+os.path.basename(fname).split('_')[0] + f"-{i}.mat"), data)
        


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)