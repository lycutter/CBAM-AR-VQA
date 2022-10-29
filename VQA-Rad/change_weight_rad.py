import os
import torch
from torch.utils.data import DataLoader
import dataset_RAD

from tools import utils
from model.network import Net
from options import parse_args
import numpy as np
import torch.nn.functional as F
import _pickle as cPickle
from torch.nn.modules.loss import _Loss


try:
    import _pickle as pickle
except:
    import pickle
import torch.nn as nn



if __name__ == '__main__':

    args = parse_args()
    args.mima = False
    # args.RAD_dir = 'data_pathVQA/'
    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
    train_set = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
    batch_size = args.batch_size
    ans_token_dict = train_set.label_token_dict
    model = Net(args, len(train_set.label2ans), len(dictionary), ans_token_dict)

    model.cuda()
    model_dict = torch.load('saved_model/CTI7_5_1024_7583.pth', map_location='cuda:0')
    # old1 = model_dict['typeattn.w_emb.emb.weight']
    # old2 = model_dict['typeattn.q_final.tanh_gate.weight']

    new_key_list = []
    old_key_list = []
    for key in model_dict:
        if 'typeattn' in key:
            old_key = key
            key_last = old_key[8:]
            new_key = 'answer_refine' + key_last
            old_key_list.append(old_key)
            new_key_list.append(new_key)
            # model_dict[new_key] = model_dict[old_key]

    for i in range(len(new_key_list)):
        model_dict[new_key_list[i]] = model_dict.pop(old_key_list[i])

    # model_dict['answer_refine.w_emb.emb.weight'] = model_dict.pop('typeattn.w_emb.emb.weight')
    # model_dict['answer_refine.w_emb2.emb.weight'] = model_dict.pop('typeattn.w_emb2.emb.weight')
    # model_dict['token_reason.attention_block.tanh_gate.bias'] = model_dict.pop('typeattn.q_final.tanh_gate.bias')
    # model_dict['token_reason.attention_block.sigmoid_gate.weight'] = model_dict.pop('typeattn.q_final.sigmoid_gate.weight')
    # model_dict['token_reason.attention_block.sigmoid_gate.bias'] = model_dict.pop('typeattn.q_final.sigmoid_gate.bias')
    # model_dict['token_reason.attention_block.attn.weight'] = model_dict.pop('typeattn.q_final.attn.weight')
    # model_dict['token_reason.attention_block.attn.bias'] = model_dict.pop('typeattn.q_final.attn.bias')
    # model_dict['token_reason.q_emb.rnn.weight_ih_l0'] = model_dict.pop('typeattn.q_emb.rnn.weight_ih_l0')
    # model_dict['token_reason.q_emb.rnn.weight_hh_l0'] = model_dict.pop('typeattn.q_emb.rnn.weight_hh_l0')
    # model_dict['token_reason.q_emb.rnn.bias_ih_l0'] = model_dict.pop('typeattn.q_emb.rnn.bias_ih_l0')
    # model_dict['token_reason.q_emb.rnn.bias_hh_l0'] = model_dict.pop('typeattn.q_emb.rnn.bias_hh_l0')
    # model_dict['token_reason.fc_img.weight'] = model_dict.pop('typeattn.fc_img.weight')
    # model_dict['token_reason.fc_img.bias'] = model_dict.pop('typeattn.fc_img.bias')
    # model_dict['token_reason.f_fc.weight'] = model_dict.pop('typeattn.f_fc.weight')
    # model_dict['token_reason.f_fc.bias'] = model_dict.pop('typeattn.f_fc.bias')


    torch.save(model_dict, 'saved_models/vqa_rad_best.pth')

    print()


    # model.load_state_dict(torch.load('saved_models/path_bound2.pth', map_location='cuda:0'))
    # print()