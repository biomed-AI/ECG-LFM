import numpy as np
import pandas as pd 
import torch 
import torch.nn as nn
import neurokit2 as nk
from fairseq_signals.utils import checkpoint_utils, options, utils
from captum.attr import IntegratedGradients, LayerIntegratedGradients, DeepLiftShap, GuidedGradCam, LayerGradCam, visualization, LayerAttribution
import wfdb
import scipy.io
import os
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt
from torchvision import models

class ECG_dataset():
    def __init__(self, manifest_path):
        with open(manifest_path, 'r') as file:
            # 读取第一行内容
            root_path = file.readline().split('\n')[0]

        self.manifest_df = pd.read_csv(manifest_path, header=None, sep='\t', skiprows=1)
        
        self.pids = []
        self.paths = []
        for a in range(len(self.manifest_df)):
            pid = self.manifest_df.loc[a, 0].split('/')[-1].split('_')[0]
            self.pids.append(pid)
            self.paths.append(os.path.join(root_path, self.manifest_df.loc[a, 0]))
        self.len = len(self.pids)
        self.sample_rate = 500
        self.leads_to_load = list(range(12))
        self.filter = True
        self.normalize = False
        self.training = False
        self.apply_perturb = False


    def postprocess(self, feats, curr_sample_rate=None, leads_to_load=None):
        if (
            (self.sample_rate is not None and self.sample_rate > 0)
            and curr_sample_rate != self.sample_rate
        ):
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        leads_to_load = self.leads_to_load if leads_to_load is None else leads_to_load
        feats = feats.float()
        feats = feats[leads_to_load]

        if self.filter:
            feats = torch.from_numpy(
                np.stack([nk.ecg_clean(l, sampling_rate=500) for l in feats])
            )
        if self.normalize:
            for l in leads_to_load:
                feats[l] = (feats[l] - self.mean[l]) / self.std[l]

        if self.training and self.apply_perturb:
            feats = self.perturb(feats)

        return feats

    def __getitem__(self, index):
        pid = self.pids[index]
        path = self.paths[index]
        ecg = scipy.io.loadmat(path)
        feats = torch.from_numpy(ecg['feats']).float()
        label = ecg['label']
        curr_sample_rate = ecg['curr_sample_rate']
        out = {}
        out["source"] = self.postprocess(feats, curr_sample_rate, self.leads_to_load).float()
        #out["source"] = feats
        out["padding_mask"] = torch.zeros(12, 2500)
        # out["label"] = label
        out["path"] = path

        return out

    def __len__(self):
        return self.len 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/manifest2/cpsc_2018_AF/train.tsv'
manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_MI/test.tsv'
manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_CD/test.tsv'
manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_STTC/test.tsv'
manifest_path = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_HYP/test.tsv'

ECG_dataset = ECG_dataset(manifest_path)
dataloader = torch.utils.data.DataLoader(dataset=ECG_dataset, batch_size=1, num_workers=1)
# for step,(input) in enumerate(tqdm(dataloader)):
#     if input['label'][0][0][0]==1:
#         print(input["path"])
# exit()

save_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/physionet.org/files/explain/cpsc_2018_AF/test'
save_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain/ptbxl_MI/test'
save_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain/ptbxl_CD/test'
save_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain/ptbxl_STTC/test'
save_dir = '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/explain/ptbxl_HYP/test'

weight_path = '/home/linsy/ECG_FM/data/pt/checkpoint_best.pt'
weight_path = '/bigdat2/user/linsy/bigdat1/linsy/MIMIC/ECG_copy/physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/XG_output/2024.11.10/finetune/MI/checkpoint_best.pt'
weight_path = '/bigdat2/user/linsy/bigdat1/linsy/MIMIC/ECG_copy/physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/XG_output/2024.11.10/finetune/CD/checkpoint_best.pt'
weight_path = '/bigdat2/user/linsy/bigdat1/linsy/MIMIC/ECG_copy/physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/XG_output/2024.11.10/finetune/STTC/checkpoint_best.pt'
weight_path = '/bigdat2/user/linsy/bigdat1/linsy/MIMIC/ECG_copy/physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/XG_output/2024.11.10/finetune/HYP/checkpoint_best.pt'
#weight_path = '/bigdat2/user/linsy/bigdat1/linsy/MIMIC/ECG_copy/physionet.org/static/published-projects/mimic-iv-ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/XG_output/2024.11.10/checkpoint80.pt'

overrides = dict({'task': {'data': '/bigdat2/user/linsy/bigdat1/linsy/Dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/manifest_preprocessed_ptbxl_CD'}, 'model_path': None, 'no_pretrained_weights': True})
model, saved_cfg, task = checkpoint_utils.load_model_and_task(
        weight_path,
        arg_overrides=overrides,
        suffix=''
    )
model = model.to(device)
model.eval()

step_stop = 150
for step,input in enumerate(tqdm(dataloader)):
    if step==step_stop:
        exit()
    file_name = os.path.basename(input['path'][0]).split('.')[0]
    filepath = os.path.join(save_dir, file_name+'.npy')
    
    # ECG_array = input['source'].numpy()
    # save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test1_ECG_.npy'
    # np.save(save_path, ECG_array)
    # # print(model(input['source']))
    # # print(model.encoder.encoder.layers[-1].final_layer_norm)
    # #print(model.encoder.encoder.layers)

    # #model.encoder.feature_extractor.conv_layers[-1][0]
    # #model.encoder.encoder.layers[-1].conv_module.pointwise_conv1

    # guided_gc = GuidedGradCam(model, model.encoder.encoder.layers[11].conv_module.pointwise_conv1)
    # attributions = guided_gc.attribute(inputs=input['source'], target=0)
    # print(attributions.shape)
    # #attributions = LayerAttribution.interpolate(attributions, (2500))
    # attributions = attributions.squeeze(0).detach().numpy()
    # print(attributions.shape)

    # save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/gc_test1_ECG.npy'
    # np.save(save_path, attributions)
    # #visualization.visualize_image_attr(attributions)

    # attributions = np.maximum(attributions, 0)
    # attributions = attributions/np.max(attributions)
    # print(np.min(attributions), np.max(attributions))
    # save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/GradCam/gc_test1.png'
    # plt.figure()
    # plt.imshow(attributions, cmap='jet')  # 使用jet colormap进行热图可视化
    # plt.axis('off')  # 关闭坐标轴
    # plt.savefig(save_path)  # 保存为图片文件
    # exit()

    #layer_cond = LayerIntegratedGradients(model, model.encoder.encoder.layers[-1].final_layer_norm)
    layer_cond = LayerIntegratedGradients(model, model.encoder.encoder.layers[-1].ffn2.w_2)
    attributions = layer_cond.attribute(inputs=input['source'], target=0)

    attribution_array = attributions.transpose(0, 1).numpy()[0]
    npy_data = np.mean(np.abs(attribution_array),0)
    #npy_data = (npy_data - np.min(npy_data)) / (np.max(npy_data) - np.min(npy_data))
    top_indices = sorted(range(len(npy_data)), key=lambda i: npy_data[i], reverse=True)[:5]
    print("最大的五个特征的索引为:", top_indices)

    np.save(filepath, npy_data)
    #print(npy_data, np.max(npy_data), np.min(npy_data), list(npy_data).index(np.max(npy_data)), npy_data.shape)
    continue

    ig = IntegratedGradients(model)
    baseline_source = torch.rand(1, 12, 2500, requires_grad=True).to(device)
    baseline_padding_mask = torch.zeros(1, 12, 2500) 
    attributions = ig.attribute(inputs=input['source'], target=0)
    attribution_array = attributions.numpy()
    print(attribution_array)
    save_path = '/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test3.npy'
    np.save(save_path, attribution_array)


    exit()

    baseline = {}
    baseline['source'] = baseline_source
    baseline['padding_mask'] = baseline_padding_mask

    print(model(baseline_source))
    inputs = input['source']
    attributions_source = ig.attribute(inputs = input['source'], target=0)
    print("Source Input Attribution:", attributions_source)

    print(model(**input))
    exit()

source = torch.rand(1, 12, 2500)
padding_mask = torch.zeros(1, 12, 2500, dtype=torch.bool)
sample = {}
sample['source'] = source
sample['padding_mask'] = padding_mask
print(model(**ECG_dataset[0]))

