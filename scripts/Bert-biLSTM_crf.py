# coding: utf-8

import sys, os, time, gc, json, logging
import torch
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, BertModel

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example import Example4Bert, SLUDataset
from utils.evaluator import Evaluator
from utils.batch import from_example_list
from utils.vocab import PAD
from utils.utils import setup_logger
from model.Bert_biLSTM_crf import Model


# def set_optimizer(model, args):
#     params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
#     grouped_params = [{'params': list(set([p for n, p in params]))}]
#     optimizer = Adam(grouped_params, lr=args.lr)
#     return optimizer


def idx2word(input_ids, tokenizer):
    utts_list = []
    for i in range(input_ids.shape[0]):
        decode_str = tokenizer.decode(input_ids[i])
        word_list = decode_str.split(' ')
        new_word_list = []
        for word in word_list:
            if word != '[PAD]':
                new_word_list.append(word)
        utts_list.append(new_word_list)
    return utts_list


def unpadding(labels, attention_mask):
    new_labels = []
    for i in range(attention_mask.shape[0]):
        mask = attention_mask[i] == 1
        new_labels.append(labels[i][mask].cpu().numpy().tolist())
    return new_labels


def decode(args, choice, dataloader, model, tokenizer, device):
    assert choice in ['train', 'dev']
    model.eval()
    evaluator = Evaluator()
    # dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, pred_labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i, (input, labels) in enumerate(dataloader):
            input = input.to(device)
            labels = labels.to(device)
            output, loss = model(input, labels)
            pred = model.decode(label_vocab, output, idx2word(input['input_ids'], tokenizer))
            unpadded_label_ids = unpadding(labels, input['attention_mask'])
            unpadded_label = model.decode(label_vocab, unpadded_label_ids, idx2word(input['input_ids'], tokenizer))
            predictions.extend(pred)
            pred_labels.extend(unpadded_label)
            total_loss += loss.item()
        count += 1
        metrics = evaluator.acc(predictions, pred_labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict(args, test_path, model, tokenizer, device):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for i, (input, labels) in enumerate(train_dataloader):
            input = input.to(device)
            labels = labels.to(device)
            output, _ = model(input, labels)
            pred = model.decode(label_vocab, output, idx2word(input['input_ids'], tokenizer))
            for pi, p in enumerate(pred):
                predictions[pi] = p
    test_json = json.load(open(test_path, 'r'))
    ptr = 0
    print(predictions)
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[ei]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


# initialization params, output path, logger, random seed and torch.device
if __name__ == "__main__":
    args = init_args(sys.argv[1:])
    logdir = os.path.join('./logs/Bert-CRF', args.sub_logdir)

    if (not os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = setup_logger('logger', os.path.join(logdir, 'train.log'))

    set_random_seed(args.seed)
    device = set_torch_device(args.device)
    logger.info("Initialization finished ...")
    logger.info("Random seed is set to %d" % (args.seed))
    logger.info("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

    start_time = time.time()
    train_path = os.path.join(args.dataroot, 'train.json')
    dev_path = os.path.join(args.dataroot, 'development.json')
    test_path = os.path.join(args.dataroot, 'test_llm_subset.json')
    # Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
    Example4Bert.configuration(args.dataroot)
    train_dataset = SLUDataset(train_path)
    dev_dataset = SLUDataset(dev_path)
    test_dataset = SLUDataset(test_path)
    logger.info("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
    logger.info("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

    pretrained_checkpoint = '/data/jiude/LM/model/pretrained_model'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, cache_dir=None, force_download=False)
    word_vocab = tokenizer.get_vocab()
    args.vocab_size = len(word_vocab)
    args.pad_idx = 0
    args.num_tags = Example4Bert.label_vocab.num_tags + 1
    # args.num_tags = Example4Bert.label_vocab.num_tags
    args.tag_pad_idx = Example4Bert.label_vocab.convert_tag_to_idx(PAD)

    label_vocab = Example4Bert.label_vocab
    label_vocab.idx2tag[74] = '[SPE-TOKEN]'
    label_vocab.tag2idx['[SPE-TOKEN]'] = 74

    Bert = AutoModel.from_pretrained(pretrained_checkpoint,
                                    id2label = label_vocab.idx2tag,
                                    label2id = label_vocab.tag2idx).to(device)
    

    def collate_fn(data):
        utt_batch = [item['utt'] for item in data]
        label_batch = [item['labels'] for item in data]
        inputs = tokenizer.batch_encode_plus(utt_batch, padding=True, return_tensors='pt')

        batch_len = inputs['input_ids'].shape[1]
        for i in range(len(label_batch)):
            label_batch[i] = [74] + label_batch[i]
            label_batch[i] += [74] * batch_len
            label_batch[i] = label_batch[i][:batch_len]

        return inputs, torch.tensor(label_batch, dtype=torch.long)
    

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                                   collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=args.batch_size, 
                                                 collate_fn=collate_fn, shuffle=False, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, 
                                                  collate_fn=collate_fn, shuffle=True, drop_last=True)


    model = Model(args.num_tags, args.batch_size, Bert, args.finetune).to(device)
    # Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

    if args.testing:
        check_point = torch.load(open(os.path.join(logdir, 'model.bin'), 'rb'), map_location=device)
        model.load_state_dict(check_point['model'])
        logger.info("Load saved model from root path")


    if not args.testing:
        num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
        logger.info('Total training steps: %d' % (num_training_steps))
        # optimizer = set_optimizer(model, args)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        best_result = {'dev_acc': 0., 'dev_f1': 0.}
        logger.info('Start training ......')
        for i in range(args.max_epoch):
            start_time = time.time()
            epoch_loss = 0
            model.train()
            count = 0
            for j, (input, labels) in enumerate(train_dataloader):
                input = input.to(device)
                labels = labels.to(device)
                output, loss = model(input, labels)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                count += 1
            logger.info('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
            torch.cuda.empty_cache()
            gc.collect()

            start_time = time.time()
            metrics, dev_loss = decode(args, 'dev', dev_dataloader, model, tokenizer, device)
            dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
            logger.info('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
            if dev_acc > best_result['dev_acc']:
                best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
                torch.save({
                    'epoch': i, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, open(os.path.join(logdir, 'model.bin'), 'wb'))
                logger.info('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

        logger.info('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
    else:
        start_time = time.time()
        metrics, dev_loss = decode(args, 'dev', dev_dataloader, model, tokenizer, device)
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        predict(args, test_path, model, tokenizer, device)
        logger.info("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
