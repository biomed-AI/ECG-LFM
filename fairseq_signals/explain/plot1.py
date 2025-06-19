import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import scipy.io

def RGB_to_Hex(rgb):
    RGB = rgb.split(',')            # 将RGB格式划分开来
    color = '#'
    for i in RGB:
        num = int(i)
        # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color
    
def print_wav(ecg_data, importance):

    # data_points = ecg_data[0,:]
    # importance_data = importance[0,:]
    # # 将重要性数据分成几个段落
    # num_segments = 1000
    # segmented_importance = np.digitize(importance_data, bins=np.linspace(0, 1, num_segments))

    # # 创建一个颜色映射，用于为不同段落设置不同的颜色
    # colors = plt.cm.coolwarm(np.linspace(0, 1, num_segments))
    # colors = sns.color_palette("coolwarm", num_segments)

    # # 绘制折线图，根据分段的重要性数据调整段落的颜色
    # plt.figure(figsize=(12, 6))
    # prev_segment = segmented_importance[0]
    # for i in range(1, len(data_points)):
    #     if segmented_importance[i] != prev_segment:
    #         plt.plot(range(i-1, i+1), data_points[i-1:i+1], color=colors[prev_segment], linewidth=2)
    #         prev_segment = segmented_importance[i]

    # plt.title('Line Plot with Segmented Importance')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test.png')  # 保存为PNG格式图片
    # plt.show()
    # exit()

    # 假设数据是采样率为1000Hz的，您可能需要根据实际情况调整
    fs = 1  # 采样率

    # 绘制心电图
    plt.figure(figsize=(10, 7))
    t = np.arange(0, ecg_data.shape[1]) / fs  # 时间轴
    print(t.shape)
    title_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for i in tqdm(range(12)):
        plt.subplot(12, 1, i+1)
        data_points = ecg_data[i,:]
        importance_data = importance[i,:]
        num_segments = 1000
        segmented_importance = np.digitize(importance_data, bins=np.linspace(0, 1, num_segments))
        colors = sns.color_palette("rocket_r", num_segments+1) #rocket_r
        prev_segment = segmented_importance[0]
        for j in range(1, len(data_points)):
            #print(segmented_importance[j], prev_segment)
            if segmented_importance[j] != prev_segment:
                
                plt.plot(range(j-1, j+1), data_points[j-1:j+1], color=colors[prev_segment], linewidth=2)
                prev_segment = segmented_importance[j]

        # segmented_importance = np.digitize(importance_data, bins=np.linspace(0, 1, 10))
        # df = pd.DataFrame({'x': t, 'y': ecg_data[i,:], 'importance': segmented_importance})

        #sns.lineplot(x='x', y='y', data=df, palette=sns.color_palette("coolwarm", n_colors=10), hue='importance', linewidth=2)
        #plt.plot(t, ecg_data[i,:], c=importance[i,:], cmap='coolwarm')

        plt.title(title_list[i])
        plt.xlabel('Time (ms)')
        plt.ylabel('')

    plt.tight_layout()

    # 保存为图片
    plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test3_ECG.png')  # 保存为PNG格式图片
    plt.show()

def plot_ig_top5_fea():
    ig_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/explain/cpsc_2018_AF/test'
    ig_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain/ptbxl_STTC/test'
    all_ig_fea = []
    for file_name in tqdm(list(os.listdir(ig_dir))[0:300]):
        filepath = os.path.join(ig_dir, file_name)
        ig_fea = np.load(filepath)
        all_ig_fea.append(ig_fea)
    IG_npy = np.array(all_ig_fea)
    IG_npy = np.mean(np.abs(IG_npy),0)
    IG_npy = IG_npy/np.sum(IG_npy)

    ECG_shortname_list = ['fea'+str(a) for a in range(0,1024)]

    top_10_indices = [ECG_shortname_list.index(feature) for feature in ECG_shortname_list]
    top_10_indices = sorted(top_10_indices, key=lambda x: IG_npy[x], reverse=True)[:10]
    for idx in top_10_indices:
        print(f"Feature Name: {ECG_shortname_list[idx]}, Importance Score: {IG_npy[idx]}")
    save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/plot/explain/ptbxl_STTC_ig_top5_fea_bar.png'

    data = {'Feature Name': [ECG_shortname_list[idx] for idx in top_10_indices],
        'Importance Score': [IG_npy[idx] for idx in top_10_indices]}
    result_df = pd.DataFrame(data)

    plt.figure(figsize=(5,3.5))
    ax = plt.subplot()
    PCC_fig = sns.barplot(x='Feature Name', y='Importance Score', data=result_df, color='#3098F8')
    PCC_fig = PCC_fig.get_figure()
    #ax.set(ylim=(0.5, 0.85))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    handles, labels = ax.get_legend_handles_labels()
    #plt.legend(handles[0:5], labels[0:5], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.tight_layout()
    PCC_fig.savefig(save_path, dpi = 400)

def plot_ig_top5_fea_unnormal():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_CD/test.tsv'
    with open(manifest_path, 'r') as file:
        # 读取第一行内容
        root_path = file.readline().split('\n')[0]

    ig_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/explain/cpsc_2018_AF/test'
    ig_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain/ptbxl_CD/test'
    all_ig_fea = []
    for file_name in tqdm(list(os.listdir(ig_dir))[0:300]):
        filepath = os.path.join(ig_dir, file_name)
        mat_path = os.path.join(root_path, os.path.basename(filepath).split('.')[0]+'.mat')
        ecg = scipy.io.loadmat(mat_path)
        label = ecg['label'][0][0]
        if label:
            ig_fea = np.load(filepath)
            all_ig_fea.append(ig_fea)

    IG_npy = np.array(all_ig_fea)
    IG_npy = np.mean(np.abs(IG_npy),0)
    IG_npy = IG_npy/np.sum(IG_npy)

    ECG_shortname_list = ['fea'+str(a) for a in range(0,1024)]

    top_10_indices = [ECG_shortname_list.index(feature) for feature in ECG_shortname_list]
    top_10_indices = sorted(top_10_indices, key=lambda x: IG_npy[x], reverse=True)[:10]
    for idx in top_10_indices:
        print(f"Feature Name: {ECG_shortname_list[idx]}, Importance Score: {IG_npy[idx]}")
    save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/plot/explain/ptbxl_CD_unnormal_ig_top5_fea_bar.png'

    data = {'Feature Name': [ECG_shortname_list[idx] for idx in top_10_indices],
        'Importance Score': [IG_npy[idx] for idx in top_10_indices]}
    result_df = pd.DataFrame(data)

    plt.figure(figsize=(5,3.5))
    ax = plt.subplot()
    PCC_fig = sns.barplot(x='Feature Name', y='Importance Score', data=result_df, color='#3098F8')
    PCC_fig = PCC_fig.get_figure()
    #ax.set(ylim=(0.5, 0.85))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    handles, labels = ax.get_legend_handles_labels()
    #plt.legend(handles[0:5], labels[0:5], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.tight_layout()
    PCC_fig.savefig(save_path, dpi = 400)

def plot_ig_top5_fea_unnormal_linear_prob():
    manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_CD/train.tsv'
    with open(manifest_path, 'r') as file:
        # 读取第一行内容
        root_path = file.readline().split('\n')[0]

    ig_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain_linear_prob/ptbxl_STTC'
    df_save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/EDF_importance/ptbxl_STTC.csv'
    save_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/plot/explain/ptbxl_STTC_unnormal_linear_prob_ig_top5_fea_bar.png'

    all_ig_fea = []
    for file_name in tqdm(list(os.listdir(ig_dir))[1000:3000]):
        filepath = os.path.join(ig_dir, file_name)
        mat_path = os.path.join(root_path, os.path.basename(filepath).split('.')[0]+'.mat')
        ecg = scipy.io.loadmat(mat_path)
        label = ecg['label'][0][0]
        if label:
            ig_fea = np.load(filepath)
            all_ig_fea.append(ig_fea)

    IG_npy = np.array(all_ig_fea)
    IG_npy = np.mean(np.abs(IG_npy),0)
    IG_npy = IG_npy/np.sum(IG_npy)

    ECG_shortname_list = ['fea'+str(a) for a in range(0,1024)]

    top_10_indices = [ECG_shortname_list.index(feature) for feature in ECG_shortname_list]
    top_10_indices = sorted(top_10_indices, key=lambda x: IG_npy[x], reverse=True)[:10]
    for idx in top_10_indices:
        print(f"Feature Name: {ECG_shortname_list[idx]}, Importance Score: {IG_npy[idx]}")

    data = {'Feature Name': [ECG_shortname_list[idx] for idx in top_10_indices],
        'Importance Score': [IG_npy[idx] for idx in top_10_indices]}
    result_df = pd.DataFrame(data)

    result_df.to_csv(df_save_path, index=False)

    plt.figure(figsize=(5,3.5))
    ax = plt.subplot()
    PCC_fig = sns.barplot(x='Feature Name', y='Importance Score', data=result_df, color='#3098F8')
    PCC_fig = PCC_fig.get_figure()
    #ax.set(ylim=(0.5, 0.85))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    handles, labels = ax.get_legend_handles_labels()
    #plt.legend(handles[0:5], labels[0:5], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    plt.tight_layout()
    PCC_fig.savefig(save_path, dpi = 400)

def ig_top5_fea_plot():
    df_dir = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/data/EDF_importance'
    png_dir = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/plot/EDF_importance'
    CVD_list = ['MI', 'HYP', 'CD', 'STTC', 'AFIB', 'GSVT', 'SB', 'VE']
    for CVD in CVD_list:
        df_path = os.path.join(df_dir, 'ptbxl_'+CVD+'2.csv')
        df = pd.read_csv(df_path)

        save_path = os.path.join(png_dir, CVD+'_EDF_importance.png')
        plt.figure(figsize=(3.5,3.5))
        ax = plt.subplot()
        PCC_fig = sns.barplot(x='Feature Name', y='Importance Score', data=df, color='#3098F8')
        PCC_fig = PCC_fig.get_figure()
        #ax.set(ylim=(0.5, 0.85))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Importance score")
        handles, labels = ax.get_legend_handles_labels()
        #plt.legend(handles[0:5], labels[0:5], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
        plt.tight_layout()
        PCC_fig.savefig(save_path, dpi = 400)

def identification_barplot():
    result_path = '/bigdat2/user/linsy/bigdat1/linsy/ECG_FM/plot/pretrain/identification.xlsx'
    result_df = pd.read_excel(result_path)
    

identification_barplot()
exit()
npy_path = '/home/linsy/ECG_FM/code/ECG_CNN/data/gc_test1_ECG.npy'
npy_data = np.load(npy_path)
#npy_data = np.mean(npy_data, 0).reshape(1,-1)

npy_data = np.abs(npy_data)
#npy_data = (npy_data - npy_data.min(axis=1, keepdims=True)) / (npy_data.max(axis=1, keepdims=True) - npy_data.min(axis=1, keepdims=True))
npy_data = (npy_data - np.min(npy_data)) / (np.max(npy_data)*1 - np.min(npy_data))
data = npy_data
print(np.max(npy_data))
save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/GradCam/gc_test2.png'
vmin_color = 'blue'
vmax_color = 'red'

# 绘制热力图，并设定最大值和最小值对应的颜色
plt.figure()
sns.heatmap(data, cmap='rocket_r', vmin=np.min(npy_data), vmax=np.max(npy_data), cbar_kws={'label': 'Colorbar'})
plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/GradCam/gc_test2.png')
exit()


ECG_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test3_ECG.npy'
ECG_data = np.load(ECG_path)[0]
print_wav(ECG_data, npy_data)

# plt.figure(figsize=(12, 6))
# plt.imshow(npy_data, cmap='coolwarm', interpolation='nearest', aspect='auto')
# plt.colorbar()
# plt.title('Heatmap of ECG Feature Importance')
# plt.xlabel('Time')
# plt.ylabel('Lead')

#plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test1.png')