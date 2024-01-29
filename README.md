# BERT-CRF: Model for Chinese Spoken Language Understanding

<center>Project of CS3602 NLP, 2023 Fall, SJTU</center>

<center> [Jiude Wei](https://github.com/kizunawl) [Letian Yang](https://github.com/MoeKid101) </center>

## Requirements
    conda create -n nlp python=3.7
    conda activate nlp
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install transformers
    pip install tqdm

## Training

Please refer to our report in this repository for more details of this project. 

Please refer to [bert-base-chinese](https://huggingface.co/bert-base-chinese/tree/main) for pretrained model we use in this task. 

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
