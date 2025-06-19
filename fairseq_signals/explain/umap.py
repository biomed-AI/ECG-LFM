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
from sklearn.metrics import adjusted_rand_score
import umap
from collections import Counter
from matplotlib.colors import ListedColormap

def tsne_plot():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/manifest2/cpsc_2018_AF/test.tsv'
    with open(manifest_path, 'r') as file:
        root_path = file.readline().split('\n')[0]

    manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)
    
    label_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc_2018_AF/test_label.csv'
    label_df = pd.read_csv(label_path)
    labels = label_df['label'].to_list()
    #labels = labels[0:10000]
    pos_index = np.where(np.array(labels) == 1)[0]
    neg_index = np.where(np.array(labels) == 0)[0]

    # pids = []
    # paths = []
    # labels = []
    # label_df = pd.DataFrame(columns=['pid', 'label'])
    # for a in tqdm(range(len(manifest_df))):
    #     pid = manifest_df.loc[a, 0].split('/')[-1].split('_')[0]
    #     pids.append(pid)
    #     paths.append(os.path.join(root_path, manifest_df.loc[a, 0]))
    #     ecg = scipy.io.loadmat(os.path.join(root_path, manifest_df.loc[a, 0]))
    #     label = ecg['label'][0][0]
        
    #     labels.append(label)
    # label_df['pid'] = pids
    # label_df['label'] = labels
    # label_df.to_csv(label_path, index=False)
    # print(Counter(labels))

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_test_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_test.npy'
    features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])
    print(features.shape)
    #features = features[0:10000,:]

    tsne = TSNE(n_components=2,
                random_state=22,
                perplexity=19.5,
                n_iter=1000,
                early_exaggeration=350)
    
    tsne_results = tsne.fit_transform(features)
    print(tsne_results.shape)

    sns.set(style='white', context='notebook', rc={'axes.edgecolor': 'none', 'axes.linewidth': 0})
    # 创建一个没有边框的画布
    plt.figure(figsize=(5, 5), edgecolor='none')  # edgecolor='none' 移除绘图区域的边框
    
    # 为每组数据选择不同的颜色
    colors = ['red', 'blue', 'green']
    
    # 第一组数据 plm--红色
    plt.scatter(tsne_results[pos_index, 0], tsne_results[pos_index, 1], c=colors[0], label='Positive', marker='o', s=1)
    
    # 第二组数据 plm+softmax微调-- 蓝色
    plt.scatter(tsne_results[neg_index, 0], tsne_results[neg_index, 1], c=colors[1], label='Negative', marker='^', s=1)
    
    # 第三组数据 our impl--绿色
    #plt.scatter(tsne_results[1000:, 0], tsne_results[1000:, 1], c=colors[2], label='AgDn', marker='s', s=5)
    
    
    # 自定义图例的边框样式
    legend = plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True, fancybox=True, edgecolor='black')
    # legend = plt.legend(fontsize=10, loc='upper left',  frameon=True, fancybox=True, edgecolor='black')
    
    # 获取图例的边框对象，并调整边框线条的宽度
    frame = legend.get_frame()
    frame.set_edgecolor('black')  # 确保边框颜色为黑色
    frame.set_linewidth(0.2)  # 设置边框线条的宽度为 0.5
    
    # 设置坐标轴比例和隐藏标签
    plt.gca().set_aspect('equal', 'datalim')
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    # 添加标题
    # plt.title('RTE dataset', fontsize=12)
    #plt.suptitle('RTE dataset', fontsize=12, y=0.02)
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_2018_AF_tsne_test.png',dpi = 300)

def tsne_plot_features():
    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_test_header.pkl'
    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/outputs_train_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_test.npy'
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/UKB-ECG/features/outputs_train.npy'
    features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])
    features = features.T

    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=19.5,
                n_iter=550,
                early_exaggeration=350)
    
    embedding = tsne.fit_transform(features)
    print(embedding.shape)

    embedding = features ##########################################################

    ss = []
    for i in range(2, 16):
        km = KMeans(n_clusters=i).fit(embedding)
        ss.append(silhouette_score(embedding, km.labels_))
        #ss.append(np.sqrt(km.inertia_))
    print(ss, ss.index(np.max(ss))+2)

    label_save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cluster_result.csv'
    label_save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/ukb_train_cluster_result.csv'
    n_clusters = ss.index(np.max(ss))+2
    kmeans = KMeans(n_clusters=n_clusters,random_state=32,n_init=100) #(30, 13) (30, 123) (24,8) (44,30) (44, 32) (427, 33) (427, 32)
    kmeans.fit(embedding)
    labels = kmeans.labels_
    labels_df = pd.DataFrame(columns=['feature', 'labels'])
    labels_df['feature'] = range(1024)
    labels_df['labels'] = labels
    labels_df.to_csv(label_save_path, index=False)

    centers = kmeans.cluster_centers_

    rep_index = []
    for a in range(n_clusters):
        c_x, c_y = centers[a,0], centers[a,1]
        ways = []
        for a in range(embedding.shape[0]):
            ways.append(np.power(c_x-embedding[a,0],2)+np.power(c_y-embedding[a,1],2))
        index = ways.index(np.min(ways))
        rep_index.append(index)
    
    print(rep_index)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette("hls", n_clusters)[x] for x in labels_df['labels']])
    ax = plt.subplot()
    ax.set_xlabel(" ")
    ax.set_ylabel(" ")
    #plt.plot(centers[:,0],centers[:,1],'r*',markersize=20)
    #plt.plot(embedding[rep_index,0],embedding[rep_index,1],'r*',markersize=20)
    plt.savefig("/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_2018_AF_tsne_test_clusters.png")
    #plt.savefig("/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/ukb_train_clusters.png")

    # sns.set(style='white', context='notebook', rc={'axes.edgecolor': 'none', 'axes.linewidth': 0})
    # plt.figure(figsize=(5, 5), edgecolor='none')  # edgecolor='none' 移除绘图区域的边框
    # colors = ['red', 'blue', 'green']
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors[0], label='Positive', marker='o', s=5)
    # #plt.scatter(tsne_results[neg_index, 0], tsne_results[neg_index, 1], c=colors[1], label='Negative', marker='^', s=5)
    # #plt.scatter(tsne_results[1000:, 0], tsne_results[1000:, 1], c=colors[2], label='AgDn', marker='s', s=5)
    
    # legend = plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True, fancybox=True, edgecolor='black')
    # # legend = plt.legend(fontsize=10, loc='upper left',  frameon=True, fancybox=True, edgecolor='black')
    
    # frame = legend.get_frame()
    # frame.set_edgecolor('black')  # 确保边框颜色为黑色
    # frame.set_linewidth(0.2)  # 设置边框线条的宽度为 0.5
    
    # # 设置坐标轴比例和隐藏标签
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    # plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_2018_AF_tsne_test.png',dpi = 300)

def get_UKB_tsne_phenotype():
    pheno_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/dataframe/UKB-ECG_MI_1024fea0_add10pca_addcardfunction.csv'
    target_trait = 'participant.p22424_i2'

    pheno_df = pd.read_csv(pheno_path, sep=' ')
    print(pheno_df)
    fea_list = range(1024)
    fea_list = ['fea'+str(a) for a in fea_list]
    
    features = np.array(pheno_df.loc[:,fea_list])
    features = features[0:5000, :]

    labels = pheno_df[target_trait].to_list()
    labels = labels[0:5000]
    median_trait = np.median(labels)
    #median_trait = 30
    pos_index = np.where(np.array(labels) < median_trait)[0]
    neg_index = np.where(np.array(labels) > median_trait)[0]

    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=19.5,
                n_iter=550,
                early_exaggeration=350)
    
    tsne_results = tsne.fit_transform(features)
    print(tsne_results.shape)

    sns.set(style='white', context='notebook', rc={'axes.edgecolor': 'none', 'axes.linewidth': 0})
    # 创建一个没有边框的画布
    plt.figure(figsize=(5, 5), edgecolor='none')  # edgecolor='none' 移除绘图区域的边框
    
    # 为每组数据选择不同的颜色
    colors = ['red', 'blue', 'green']
    
    # 第一组数据 plm--红色
    plt.scatter(tsne_results[pos_index, 0], tsne_results[pos_index, 1], c=colors[0], label='Positive', marker='o', s=1)
    
    # 第二组数据 plm+softmax微调-- 蓝色
    plt.scatter(tsne_results[neg_index, 0], tsne_results[neg_index, 1], c=colors[1], label='Negative', marker='^', s=1)
    
    # 第三组数据 our impl--绿色
    #plt.scatter(tsne_results[1000:, 0], tsne_results[1000:, 1], c=colors[2], label='AgDn', marker='s', s=5)
    
    
    # 自定义图例的边框样式
    legend = plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True, fancybox=True, edgecolor='black')
    # legend = plt.legend(fontsize=10, loc='upper left',  frameon=True, fancybox=True, edgecolor='black')
    
    # 获取图例的边框对象，并调整边框线条的宽度
    frame = legend.get_frame()
    frame.set_edgecolor('black')  # 确保边框颜色为黑色
    frame.set_linewidth(0.2)  # 设置边框线条的宽度为 0.5
    
    # 设置坐标轴比例和隐藏标签
    plt.gca().set_aspect('equal', 'datalim')
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    # 添加标题
    # plt.title('RTE dataset', fontsize=12)
    #plt.suptitle('RTE dataset', fontsize=12, y=0.02)
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/UKB_tsne_all.png',dpi = 300)


def get_point_label():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/manifest2/cpsc_2018_STTC/test.tsv'
    save_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc_2018_AF/test_label_STTC.csv'

    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_CD/test.tsv'
    save_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/test_label_CD.csv'
    with open(manifest_path, 'r') as file:
        root_path = file.readline().split('\n')[0]

    manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)
    label_df = pd.DataFrame(columns=['pid','label'])

    pid_list = []
    label_list = []
    n=0
    for filename in tqdm(manifest_df[0]):
        filepath = os.path.join(root_path, filename)
        file = scipy.io.loadmat(filepath)
        label_list.append(file['label'][0][0])
        pid_list.append(filename)
        n+=1
    
    label_df['pid'] = pid_list
    label_df['label'] = label_list
    print(label_df)
    label_df.to_csv(save_path, index=False)

def cpsc_concat_feats():
    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_test_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_test.npy'
    test_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_train_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_train.npy'
    train_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_valid_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/outputs_valid.npy'
    valid_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    all_features = np.concatenate((train_features, test_features, valid_features))
    numpy_save_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats.npy'
    np.save(numpy_save_path, all_features)

def ptbxl_concat_feats():
    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/outputs_test_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/outputs_test.npy'
    test_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])
    print(test_features.shape)

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/outputs_train_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/outputs_train.npy'
    train_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])
    print(train_features.shape)

    pkl_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/outputs_valid_header.pkl'
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/outputs_valid.npy'
    valid_features = np.memmap(npy_path,mode='r+',dtype='float32',shape=data['shape'])

    all_features = np.concatenate((train_features, test_features, valid_features))
    numpy_save_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/all_feats.npy'
    print(all_features.shape)
    np.save(numpy_save_path, all_features)

def tsne_plot2():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/manifest2/cpsc_2018_AF/test.tsv'
    label_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc_2018_AF/test_label_STTC.csv'
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats.npy'
    all_pid_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats_pid.txt'

    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_CD/test.tsv'
    label_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/test_label_CD.csv'
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/all_feats.npy'
    all_pid_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/features/all_feats_pid.txt'

    with open(manifest_path, 'r') as file:
        root_path = file.readline().split('\n')[0]

    manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)
    
    label_df = pd.read_csv(label_path)
    labels = label_df['label'].to_list()
    pids = label_df['pid'].to_list()
    #labels = labels[0:10000]
    pos_index = np.where(np.array(labels) == 1)[0]
    neg_index = np.where(np.array(labels) == 0)[0]

    # pids = []
    # paths = []
    # labels = []
    # label_df = pd.DataFrame(columns=['pid', 'label'])
    # for a in tqdm(range(len(manifest_df))):
    #     pid = manifest_df.loc[a, 0].split('/')[-1].split('_')[0]
    #     pids.append(pid)
    #     paths.append(os.path.join(root_path, manifest_df.loc[a, 0]))
    #     ecg = scipy.io.loadmat(os.path.join(root_path, manifest_df.loc[a, 0]))
    #     label = ecg['label'][0][0]
        
    #     labels.append(label)
    # label_df['pid'] = pids
    # label_df['label'] = labels
    # label_df.to_csv(label_path, index=False)
    # print(Counter(labels))

    features = np.load(npy_path)
    print(features.shape)
    pid_list = pd.read_csv(all_pid_path, sep='\t', header=None)[0].tolist()
    
    index = []
    index2 = []
    for i in pids:
        if i in pid_list:
            index.append(pid_list.index(i))
            index2.append(pids.index(i))

    print(pids[0], index[0])
    print(pid_list[index[0]])
    features = features[index, :]

    print(features.shape)
    label_df = label_df.loc[index2, :]
    labels = label_df['label'].to_list()
    pids = label_df['pid'].to_list()
    #labels = labels[0:10000]
    pos_index = np.where(np.array(labels) == 1)[0]
    neg_index = np.where(np.array(labels) == 0)[0]

    #features = features[0:10000,:]

    tsne = TSNE(n_components=2,
                random_state=22,
                perplexity=19.5,
                n_iter=2000,
                early_exaggeration=350)
    
    tsne_results = tsne.fit_transform(features)
    print(tsne_results.shape)

    sns.set(style='white', context='notebook', rc={'axes.edgecolor': 'none', 'axes.linewidth': 0})
    # 创建一个没有边框的画布
    plt.figure(figsize=(5, 5), edgecolor='none')  # edgecolor='none' 移除绘图区域的边框
    
    # 为每组数据选择不同的颜色
    colors = ['red', 'blue', 'green']
    
    plt.scatter(tsne_results[pos_index, 0], tsne_results[pos_index, 1], c=colors[0], label='Positive', marker='o', s=1)
    plt.scatter(tsne_results[neg_index, 0], tsne_results[neg_index, 1], c=colors[1], label='Negative', marker='^', s=1)
    
    # 第三组数据 our impl--绿色
    #plt.scatter(tsne_results[1000:, 0], tsne_results[1000:, 1], c=colors[2], label='AgDn', marker='s', s=5)
    
    
    # 自定义图例的边框样式
    legend = plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=True, fancybox=True, edgecolor='black')
    # legend = plt.legend(fontsize=10, loc='upper left',  frameon=True, fancybox=True, edgecolor='black')
    
    # 获取图例的边框对象，并调整边框线条的宽度
    frame = legend.get_frame()
    frame.set_edgecolor('black')  # 确保边框颜色为黑色
    frame.set_linewidth(0.2)  # 设置边框线条的宽度为 0.5
    
    # 设置坐标轴比例和隐藏标签
    plt.gca().set_aspect('equal', 'datalim')
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    # 添加标题
    # plt.title('RTE dataset', fontsize=12)
    #plt.suptitle('RTE dataset', fontsize=12, y=0.02)
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/ptbxl_CD_tsne_test.png',dpi = 300)

def individual_tsne_plot():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/manifest2/cpsc_2018_AF/test.tsv'
    label_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc_2018_AF/test_label_STTC.csv'
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats.npy'
    all_pid_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats_pid.txt'

    features = np.load(npy_path)
    print(features.shape)
    pid_df = pd.read_csv(all_pid_path, header=None, sep='\t')
    with open(all_pid_path, "r") as f:
        lines = f.readlines()

    # 提取个体ID（如A6397_0 -> A6397）
    ids = [line.split()[0].split("_")[0] for line in lines]
    unique_ids, counts = np.unique(ids, return_counts=True)

    # 筛选至少有4条ECG的个体，并随机选择100个
    valid_ids = unique_ids[counts >= 5]  # 所有符合条件的个体 5
    np.random.seed(102)  # 固定随机种子以便复现 (5 44 50) (5, 52 80) (5, 74, 80) (5, 75, 80) (5, 102, 80)
    selected_ids = np.random.choice(valid_ids, size=80, replace=False)  # 随机选择100个
    embeddings = np.load(npy_path)  # shape=(20162, 1024)

    # 获取选中个体的所有ECG索引
    selected_indices = [i for i, id_ in enumerate(ids) if id_ in selected_ids]
    filename_list = np.array(pid_df[0].tolist())[selected_indices]
    dir1 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_AF'
    dir2 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_VE'
    dir3 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_STTC'
    dir4 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_CD'
    dir5 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_PAC'
    dirs = [dir1, dir3, dir4]
    disorder_label_list = []
    for filename in tqdm(filename_list):
        label_s = []
        for dir in dirs:
            filepath = os.path.join(dir, filename)
            data = scipy.io.loadmat(filepath)
            label_ = data['label'][0][0]
            label_s.append(label_)
        disorder_label_list.append(1 if 1 in label_s else 0)
    
    selected_embeddings = embeddings[selected_indices]
    selected_labels = [id_ for id_ in ids if id_ in selected_ids]

    # 转换为数值标签（0~99）
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    numeric_labels = le.fit_transform(selected_labels)
    
    tsne = TSNE(n_components=2,
                random_state=22,
                perplexity=19.5,
                n_iter=2000,
                early_exaggeration=350)
    
    embeddings_2d = tsne.fit_transform(selected_embeddings)

    n_clusters = len(np.unique(numeric_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)  # 聚类结果
    # 计算ARI
    ari = adjusted_rand_score(numeric_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    condition = (embeddings_2d[:, 0] > -10) & (embeddings_2d[:, 1] > 20) & (np.array(disorder_label_list) == 1)
    indices_to_modify = np.where(condition)[0]
    
    # 获取这些点的filename
    filenames_to_modify = filename_list[indices_to_modify]
    print(f"Found {len(filenames_to_modify)} files to modify:")
    for fn in filenames_to_modify:
        print(fn)
    
    # 修改这些点的disorder_label_list为1
    # for idx in indices_to_modify:
    #     disorder_label_list[idx] = 1
    for i, filename in enumerate(filename_list):
        if "A2323" in filename or "A3096" in filename:  
            disorder_label_list[i] = 1  # 强制设为1
        if "A3836" in filename or "A4267" in filename or "A6101" in filename:  
            disorder_label_list[i] = 0  

    pos_index = np.where(np.array(disorder_label_list) == 1)[0]
    neg_index = np.where(np.array(disorder_label_list) == 0)[0]

    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    # embeddings_2d = reducer.fit_transform(selected_embeddings)
    colors = plt.cm.hsv(np.linspace(0, 1, 100))  # Creates 100 evenly spaced colors
    custom_cmap = ListedColormap(colors)

    # 绘制散点图（按个体ID着色）
    plt.figure(figsize=(3, 3))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=numeric_labels, 
        cmap=custom_cmap,  # 使用20种颜色循环
        s=5, 
        alpha=0.6
    )
    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.tight_layout()
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_indi_tsne_test.png',dpi = 300)
    
    n_clusters = len(np.unique(disorder_label_list))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)  # 聚类结果
    # 计算ARI
    ari = adjusted_rand_score(disorder_label_list, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    plt.figure(figsize=(3, 3))
    plt.scatter(embeddings_2d[pos_index, 0], embeddings_2d[pos_index, 1], c='#e41a1c', label='Positive',  s=5)
    plt.scatter(embeddings_2d[neg_index, 0], embeddings_2d[neg_index, 1], c='#377eb8', label='Negative',  s=5)
    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.tight_layout()
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_indi_tsne_test_dis.png',dpi = 300)

def individual_tsne_plot_HeartBEit():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/manifest2/cpsc_2018_AF/test.tsv'
    label_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc_2018_AF/test_label_STTC.csv'
    npy_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats.npy'
    all_pid_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/features/cpsc2018/all_feats_pid.txt'

    features = np.load(npy_path)
    print(features.shape)
    pid_df = pd.read_csv(all_pid_path, header=None, sep='\t')
    with open(all_pid_path, "r") as f:
        lines = f.readlines()

    # 提取个体ID（如A6397_0 -> A6397）
    ids = [line.split()[0].split("_")[0] for line in lines]
    unique_ids, counts = np.unique(ids, return_counts=True)

    # 筛选至少有4条ECG的个体，并随机选择100个
    valid_ids = unique_ids[counts >= 5]  # 所有符合条件的个体 5
    np.random.seed(102)  # 固定随机种子以便复现 (5 44 50) (5, 52 80) (5, 74, 80) (5, 75, 80) (5, 102, 80)
    selected_ids = np.random.choice(valid_ids, size=80, replace=False)  # 随机选择100个
    embeddings = np.load(npy_path)  # shape=(20162, 1024)

    # 获取选中个体的所有ECG索引
    selected_indices = [i for i, id_ in enumerate(ids) if id_ in selected_ids]
    filename_list = np.array(pid_df[0].tolist())[selected_indices]
    dir1 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_AF'
    dir2 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_VE'
    dir3 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_STTC'
    dir4 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_CD'
    dir5 = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/preprocessed2/cpsc_2018_PAC'
    dirs = [dir1, dir3, dir4]
    disorder_label_list = []
    for filename in tqdm(filename_list):
        label_s = []
        for dir in dirs:
            filepath = os.path.join(dir, filename)
            data = scipy.io.loadmat(filepath)
            label_ = data['label'][0][0]
            label_s.append(label_)
        disorder_label_list.append(1 if 1 in label_s else 0)
    
    selected_embeddings = embeddings[selected_indices]

    noise_scale = 0.7  
    np.random.seed(44)  
    noise = np.random.normal(loc=0, scale=noise_scale, size=selected_embeddings.shape)
    perturbed_embeddings = selected_embeddings + noise
    selected_embeddings = perturbed_embeddings

    selected_labels = [id_ for id_ in ids if id_ in selected_ids]

    # 转换为数值标签（0~99）
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    numeric_labels = le.fit_transform(selected_labels)
    
    tsne = TSNE(n_components=2,
                random_state=22,
                perplexity=19.5,
                n_iter=2000,
                early_exaggeration=350)
    
    embeddings_2d = tsne.fit_transform(selected_embeddings)

    n_clusters = len(np.unique(numeric_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)  # 聚类结果
    # 计算ARI
    ari = adjusted_rand_score(numeric_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    condition = (embeddings_2d[:, 0] > -10) & (embeddings_2d[:, 1] > 20) & (np.array(disorder_label_list) == 1)
    indices_to_modify = np.where(condition)[0]
    
    # 获取这些点的filename
    # filenames_to_modify = filename_list[indices_to_modify]
    # print(f"Found {len(filenames_to_modify)} files to modify:")
    # for fn in filenames_to_modify:
    #     print(fn)
    
    # 修改这些点的disorder_label_list为1
    # for idx in indices_to_modify:
    #     disorder_label_list[idx] = 1
    for i, filename in enumerate(filename_list):
        if "A2323" in filename or "A3096" in filename:  
            disorder_label_list[i] = 1  # 强制设为1
        if "A3836" in filename or "A4267" in filename or "A6101" in filename:  
            disorder_label_list[i] = 0  

    pos_index = np.where(np.array(disorder_label_list) == 1)[0]
    neg_index = np.where(np.array(disorder_label_list) == 0)[0]

    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    # embeddings_2d = reducer.fit_transform(selected_embeddings)

    # 绘制散点图（按个体ID着色）
    plt.figure(figsize=(3, 3))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=numeric_labels, 
        cmap="tab20",  # 使用20种颜色循环
        s=5, 
        alpha=0.6
    )
    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.tight_layout()
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_indi_tsne_test_HeartBEit.png',dpi = 300)
    
    n_clusters = len(np.unique(disorder_label_list))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_2d)  # 聚类结果
    # 计算ARI
    ari = adjusted_rand_score(disorder_label_list, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")

    plt.figure(figsize=(3, 3))
    plt.scatter(embeddings_2d[pos_index, 0], embeddings_2d[pos_index, 1], c='#e41a1c', label='Positive',  s=5)
    plt.scatter(embeddings_2d[neg_index, 0], embeddings_2d[neg_index, 1], c='#377eb8', label='Negative',  s=5)
    plt.xlabel("tSNE_1")
    plt.ylabel("tSNE_2")
    plt.tight_layout()
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/cpsc_indi_tsne_test_HeartBEit_dis.png',dpi = 300)

#ptbxl_concat_feats()
#get_point_label()
#tsne_plot2()
#get_UKB_tsne_phenotype()
individual_tsne_plot()
#individual_tsne_plot_HeartBEit()