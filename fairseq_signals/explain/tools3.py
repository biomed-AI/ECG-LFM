import numpy as np
import pandas as pd 
import torch 
import torch.nn as nn
from fairseq_signals.utils import checkpoint_utils, options, utils
from captum.attr import IntegratedGradients, LayerIntegratedGradients, DeepLiftShap
import wfdb
import scipy.io
import os
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import seaborn as sns
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.api as sm
from sklearn.impute import KNNImputer

def ukb_features_analysis():
    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/MI/outputs_train_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/MI/outputs_train.npy'
    train_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/MI/outputs_valid_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/MI/outputs_valid.npy'
    valid_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/MI/outputs_test_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/MI/outputs_test.npy'
    test_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    features = np.concatenate((train_features, valid_features, test_features), axis=0)
    feature_df = pd.DataFrame(features, columns=['fea'+str(i) for i in range(1, 1025)])
    print(feature_df)

    icd_path = '/bigdat2/user/linsy/bigdat1/linsy/UKB_data/pheno/icd10_clean2.txt'
    icd_df = pd.read_csv(icd_path, sep='\t')
    ECG_trait_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/ECG168_features/ecg_phenotype.tab'
    ECG_trait_df = pd.read_csv(ECG_trait_path, sep='\t')
    ECG_trait_df = ECG_trait_df.fillna(ECG_trait_df.mean())
    # imputer = KNNImputer(n_neighbors=2)
    # ECG_trait_df = pd.DataFrame(imputer.fit_transform(ECG_trait_df), columns=ECG_trait_df.columns)
    print(ECG_trait_df)

    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/manifest_MI/train.tsv'
    train_manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/manifest_MI/valid.tsv'
    valid_manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/manifest_MI/test.tsv'
    test_manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)

    manifest_df = pd.concat([train_manifest_df, valid_manifest_df, test_manifest_df], axis=0)
    manifest_df['eid'] = [int(a.split('_')[0]) for a in manifest_df[0].tolist()]
    
    icd_manifest_df = pd.merge(icd_df, manifest_df, on='eid')
    icd_manifest_ECG_df = pd.merge(icd_manifest_df, ECG_trait_df, on='FID')

    icd_manifest_ECG_fea_df = pd.concat([icd_manifest_ECG_df, feature_df], axis=1)
    print(icd_manifest_ECG_fea_df)

    ecg_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/ECG168_features/ecg_fea_info.xlsx'
    ecg_df = pd.read_excel(ecg_path)
    
    result_save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/ECG168_features/linear_analysis_168ECG_1024feas_no1.csv'
    result_df = pd.DataFrame(columns = ['fea']+ecg_df['index'].tolist())
    for fea in tqdm(['fea'+str(i) for i in range(1, 1025)]):
        pvalue_list = []
        for ECG_fea in ecg_df['index'].tolist():
            df1 = icd_manifest_ECG_fea_df.loc[:,['FID', 'eid', 0, 'sex_x', 'age_x', fea, ECG_fea]]
            #df1['intercept'] = 1.0
            X = df1[['sex_x', 'age_x', fea]]
            y = df1[ECG_fea]

            # 使用最小二乘法拟合线性模型
            model = sm.OLS(y, X).fit()
            pvalue_list.append(str(model.pvalues.tolist()[-1]))
            
        result_df.loc[len(result_df.index)] = [fea]+pvalue_list
    
    result_df.to_csv(result_save_path, index=False)
    print(result_df)


def ECG_trait_classify():
    result_save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/ECG168_features/linear_analysis_168ECG_1024feas_no1.csv'
    result_save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/ECG168_features/linear_analysis_168ECG_1024feas.csv'
    result_df = pd.read_csv(result_save_path)

    cluster_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/ukb_train_cluster_result.csv'
    cluster_df = pd.read_csv(cluster_path)
    cluster_df['fea'] = ['fea'+str(i) for i in range(1, 1025)]
    fea2cluster = dict(zip(cluster_df['fea'], cluster_df['labels']))
    cluster_label_dict = {0:{}, 1:{}, 2:{}, 3:{}}

    ecg_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/ECG168_features/ecg_fea_info.xlsx'
    ecg_df = pd.read_excel(ecg_path)
    short2full = dict(zip(ecg_df['index'], ecg_df['ECG trait']))
    for a in range(len(result_df)):
        fea = result_df.loc[a, 'fea']
        cluster_label = fea2cluster[fea]
        for ECG_fea in ecg_df['index']:
            pvalue = result_df.loc[a,ECG_fea]
            ECG_fullname = short2full[ECG_fea]
            lead = ECG_fullname.split('_')[0]
            wave = '_'.join(ECG_fullname.split('_')[1:])

            if pvalue<(0.01/(1)):
                di = cluster_label_dict[int(cluster_label)]
                if wave not in di.keys():
                    cluster_label_dict[int(cluster_label)][wave] = 0
                cluster_label_dict[int(cluster_label)][wave]+=1
            #print(pvalue, ECG_fullname, lead, wave, fea, cluster_label)
            

    print(cluster_label_dict)



ECG_trait_classify()

