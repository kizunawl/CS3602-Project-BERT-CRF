#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF
from transformers import AutoModelForTokenClassification, BertTokenizer, AutoModel


class Model(nn.Module):
    def __init__(self, num_tags, batch_size, pretrain_model, finetune=False) -> None:
        super().__init__()
        self.pretrained_model = pretrain_model
        self.batch_size = batch_size
        self.linear = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.finetune = finetune

    def loss_fct(self, logits, labels, mask):
        return -self.crf.forward(logits, labels, mask, reduction='mean')
    
    def forward(self, input, labels=None):
        if (not self.finetune):
            with torch.no_grad():
                output = self.pretrained_model(input['input_ids'], input['attention_mask'], input['token_type_ids'])
        else:
            output = self.pretrained_model(input['input_ids'], input['attention_mask'], input['token_type_ids'])

        logits = self.linear(output[0])
        mask = input['attention_mask'].bool()
        prob = self.crf.decode(logits, mask)
        # print(logits.shape, labels.shape, mask.shape)
        if labels is not None:
            loss = self.loss_fct(logits, labels, mask)
            return prob, loss
        return (prob, )
    
    def decode(self, label_vocab, prob, utt):
        batch_size = len(prob)
        predictions = []
        for i in range(batch_size):
            pred = prob[i]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        # if len(output) == 1:
        #     return predictions
        # else:
        #     loss = output[1]
        #     return predictions, labels, loss.cpu().item()
        return predictions
        
