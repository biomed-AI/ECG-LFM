![](figures/Pipeline.png)

# ECG-LFM: A self-supervised electrocardiogram foundation model for empowering cardiovascular disease prediction and genetic factor discovery 
ECG-LFM is a foundation model for electrocardiogram (ECG) analysis. ECG-LFM was developed based on the fairseq_signals framework, which implements a collection of deep learning methods for ECG analysis.

## Installation

To reproduce **ECG-LFM**, we suggest first creating a conda environment by:

~~~shell
conda create -n ECG-LFM python=3.9
conda create -n ECG-LFM
conda activate ECG-LFM
~~~

and then run the following code to install the required package:

~~~shell
cd fairseq-signals
pip install --editable ./
~~~

### Requirements
- `PyTorch version >= 1.5.0`
- `Python version >= 3.6`

## Data pre-processing

### Pre-process

Given a directory that contains .dat files from PTB-XL:

```
python fairseq_signals/data/ecg/preprocess/preprocess_ptbxl.py \
    /path/to/ptbxl/records500/ \
```

Given a directory that contains .dat files from MIMIC-IV-ECG:

```
python fairseq_signals/data/ecg/preprocess/preprocess_mimic_iv_ecg.py \
    /path/to/MIMIC-IV-ECG/ \
    --dest /path/to/output
```

### Prepare data manifest

Given a directory that contains pre-processed data:

```
python fairseq_signals/data/ecg/preprocess/manifest.py \
    /path/to/data/ \
    --dest /path/to/manifest \
    --valid-percent $valid
```

## Pre-training

Our pre-training uses the ECGLFM_600m_librivox.yaml config.

Once the relevant configuration files have been modified to suit your needs, initiate the pretraining process through Hydra's command-line interface. The sample command below illustrates some frequently used configuration overrides:

```
FAIRSEQ_SIGNALS_ROOT="<TODO>"
MANIFEST_DIR="<TODO>"
OUTPUT_DIR="<TODO>"

fairseq-hydra-train \
    task.data=$MANIFEST_DIR \
    dataset.valid_subset=valid \
    dataset.batch_size=64 \
    dataset.num_workers=10 \
    dataset.disable_validation=false \
    distributed_training.distributed_world_size=2 \
    optimization.update_freq=[2] \
    checkpoint.save_dir=$OUTPUT_DIR \
    checkpoint.save_interval=10 \
    checkpoint.keep_last_epochs=0 \
    common.log_format=csv \
    --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/ECGLFM/config/pretraining \
    --config-name ECGLFM_600m_librivox
```

## Fine-tuning

Our fine-tuning uses the diagnosis.yaml config.

Once the relevant configuration files have been modified to suit your needs, initiate the pretraining process through Hydra's command-line interface. The sample command below illustrates some frequently used configuration overrides:

```
FAIRSEQ_SIGNALS_ROOT="<TODO>"
PRETRAINED_MODEL="<TODO>"
MANIFEST_DIR="<TODO>"
OUTPUT_DIR="<TODO>"
NUM_LABELS=$(($(wc -l < "$LABEL_DIR/label_def.csv") - 1))

fairseq-hydra-train \
    task.data=$MANIFEST_DIR \
    model.model_path=$PRETRAINED_MODEL \
    model.num_labels=$NUM_LABELS \
    optimization.lr=[1e-06] \
    optimization.max_epoch=100 \
    dataset.batch_size=256 \
    dataset.num_workers=5 \
    dataset.disable_validation=true \
    distributed_training.distributed_world_size=1 \
    distributed_training.find_unused_parameters=True \
    checkpoint.save_dir=$OUTPUT_DIR \
    checkpoint.save_interval=1 \
    checkpoint.keep_last_epochs=0 \
    common.log_format=csv \
    --config-dir $FAIRSEQ_SIGNALS_ROOT/examples/ECGLFM/config/finetuning/ \
    --config-name diagnosis
```

## Citation

If you find our codes useful, please consider citing our work:

~~~bibtex


@article{
  title={A self-supervised electrocardiogram foundation model for empowering cardiovascular disease prediction and genetic factor discovery},
  author={Siying Lin, Yuedong Yang, Huiying Zhao*},
  journal={},
  year={2025},
}
<<<<<<< HEAD
~~~
=======
~~~
>>>>>>> 40f60dd (‘update’)
