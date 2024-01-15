#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torchcrf import CRF
from transformers import AutoModelForTokenClassification, BertTokenizer, AutoModel


class Model(nn.Module):
    def __init__(self, num_tags, batch_size, pretrain_model, finetune=False, hfactor=0, device='cpu') -> None:
        super().__init__()
        self.pretrained_model = pretrain_model
        self.batch_size = batch_size
        self.num_tags = num_tags
        self.device = device
        self.linear1 = nn.Linear(768, num_tags)
        self.linear2 = nn.Linear(num_tags, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.finetune = finetune
        self.max_cache_size = 128
        self.cache = None
        self.hfactor = hfactor

    def clear_cache(self):
        self.cache = None

    def loss_fct(self, logits, labels, mask):
        return -self.crf.forward(logits, labels, mask, reduction='mean')
    
    def forward(self, input, labels=None):
        if (not self.finetune):
            with torch.no_grad():
                output = self.pretrained_model(input['input_ids'], input['attention_mask'], input['token_type_ids'])
        else:
            output = self.pretrained_model(input['input_ids'], input['attention_mask'], input['token_type_ids'])

        logits = self.linear1(output[0])
        mask = input['attention_mask'].bool()

        h = torch.zeros_like(logits)
        if (self.cache != None):
            # similarity = torch.zeros(self.batch_size, len(self.cache))
            # for i in range(len(self.cache)):
            #     similarity[:, i] = torch.sum(torch.sum(torch.mul(self.cache[i][:len(logits[0])], logits),dim=2),dim=1)
            similarity = torch.einsum('pjk,qjk->pq', logits, self.cache[:, :len(logits[0])])
            p = torch.softmax(similarity, dim=1)
            # for i in range(self.batch_size):
            #     h[i] = sum(self.cache[j][:len(logits[0])] * p[i, j] for j in range(len(self.cache)))
            h = torch.einsum('ij,jkl->ikl', p, self.cache[:, :len(logits[0])])
        
        new_logits = self.linear2(logits + h * self.hfactor)

        prob = self.crf.decode(new_logits, mask)

        # for i in range(self.batch_size):
        #     padding = torch.zeros((128-len(logits[i]), self.num_tags), device=self.device)
        #     self.cache.append(torch.concat([logits[i].detach(), padding]))
        # self.cache = self.cache[-self.cache_size:]
        padding = torch.zeros((self.batch_size, 128-len(logits[0]), self.num_tags), device=self.device)
        padded_logits = torch.concat([logits.detach(), padding], dim=1)
        if (self.cache != None):
            self.cache = torch.concat([self.cache, padded_logits], dim=0)
            self.cache = self.cache[-self.max_cache_size:]
        else:
            self.cache = padded_logits

        # print(logits.shape, labels.shape, mask.shape)
        if labels is not None:
            loss = self.loss_fct(new_logits, labels, mask)
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
        
