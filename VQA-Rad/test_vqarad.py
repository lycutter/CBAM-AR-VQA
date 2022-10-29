import json
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


try:
    import _pickle as pickle
except:
    import pickle
import torch.nn as nn




def new_evaluate(model, args):
    val_set = dataset_RAD.VQAFeatureDataset('test', args, dictionary, question_len=args.question_len)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0)

    model.load_state_dict(torch.load('saved_model/vqa_rad_best.pth', map_location='cuda:0'))
    score = 0
    model.eval()
    total = 0
    open_ended = 0  # 179
    closed_ended = 0  # 272
    close_score = 0
    open_score = 0


    with torch.no_grad():
        for i, (v, q, a, ans_type, q_types, p_type) in enumerate(val_loader):
            # if p_type[0] != "freeform":
            #     continue

            v = v.cuda()
            q = q.cuda()
            a = a.cuda().float()
            target = torch.argmax(a, dim=1)

            total += q.shape[0]
            pred, _ = model(v, q)
            # batch_score = compute_score_with_logits(pred, a.data).sum()
            pred_score = torch.argmax(pred, dim=1)

            for i in range(len(ans_type)):
                if ans_type[i] == 'OPEN':
                    open_ended += 1
                    if target[i] == pred_score[i]:
                        open_score += 1
                elif ans_type[i] == 'CLOSED':
                    closed_ended += 1
                    if target[i] == pred_score[i]:
                        close_score += 1


        score += open_score + close_score

    score = (score / total)
    open_score = (open_score / open_ended)
    close_score = (close_score / closed_ended)

    return score, open_score, close_score








if __name__ == '__main__':

    args = parse_args()


    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())
    torch.cuda.set_device(args.gpu)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
    train_set = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
    val_set = dataset_RAD.VQAFeatureDataset('test', args, dictionary, question_len=args.question_len)
    ans_token_dict = train_set.label_token_dict



    batch_size = args.batch_size

    model = Net(args, len(train_set.label2ans), len(dictionary), ans_token_dict)
    model.cuda()


    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001, [0.9, 0.999])

    # optimizer = torch.optim.Adamax(params=model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 300, 400], gamma=0.1)
    score_acc, open_score_acc, close_score_acc = new_evaluate(model, args)
    print(score_acc)
    print(open_score_acc)
    print(close_score_acc)




