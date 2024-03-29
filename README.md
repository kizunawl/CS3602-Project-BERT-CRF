# BERT-CRF: Model for Chinese Spoken Language Understanding

<p align="center">
    Project of CS3602 NLP, 2023 Fall, SJTU
</p>

<p align="center">
    <a href="https://github.com/kizunawl">Jiude Wei</a>
    &nbsp
    <a href="https://github.com/MoeKid101">Letian Yang</a>
</p>

## Requirements
    conda create -n nlp python=3.7
    conda activate nlp
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install transformers
    pip install tqdm

## Training

Please refer to our <a href="https://github.com/kizunawl/CS3602-Project-BERT-CRF/blob/main/report.pdf">report</a> in this repository for more details of this project. 

Please refer to <a href="https://huggingface.co/bert-base-chinese/tree/main">bert-base-chinese</a> for pretrained model we use in this task. 

We provide five major scripts for training & evaluation

**baseline**
    
    python scripts/slu_baseline.py --device=[your_cuda_device]
    
**BERT-encoder model**

    python scripts/Bert.py --device=[your_cuda_device]

**biLSTM-CRF-decoder model**

    python scripts/biLSTM_crf.py --device=[your_cuda_device]
   
**BERT-CRF**

    python scripts/Bert-biLSTM_crf.py --device=[your_cuda_device]

**BERT-CRF with history conversation involved**

    python scripts/Bert-biLSTM_crf_useHistory.py --device=[your_cuda_device]
