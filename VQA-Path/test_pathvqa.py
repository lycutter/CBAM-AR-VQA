import json
import os
import torch
from torch.utils.data import DataLoader
import dataset_PathVQA

from tools import utils
from model.network import Net
from options import parse_args
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score


try:
    import _pickle as pickle
except:
    import pickle
import torch.nn as nn

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='sum')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='sum')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


# VQA score computation
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    one_hots = one_hots.float()
    labels = labels.float()
    scores = (one_hots * labels)
    return scores


def new_evaluate(model, args):
    val_set = dataset_PathVQA.VQAFeatureDataset('test', args, dictionary, question_len=args.question_len)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=0)

    model.load_state_dict(torch.load('saved_model/vqa_pathvqa_best.pth', map_location='cuda:0'))
    score = 0
    model.eval()
    total = 0
    open_ended = 0  # 179
    closed_ended = 0  # 272
    number_ended = 0
    close_score = 0
    open_score = 0
    number_score = 0

    #######
    pred_list = []
    label_list = []
    pred_list_open = []
    label_list_open = []
    pred_list_close = []
    label_list_close = []
    ########

    with torch.no_grad():
        for i, (v, q, a, ans_type) in enumerate(val_loader):
            # if p_type[0] != "freeform":
            #     continue
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 3, 128, 128).cuda()
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1).cuda()
            else:
                v = v.cuda().float()
            q = q.cuda()
            a = a.cuda().float()
            target = torch.argmax(a, dim=1)

            total += q.shape[0]
            pred, _ = model(v, q)
            # batch_score = compute_score_with_logits(pred, a.data).sum()
            pred_score = torch.argmax(pred, dim=1)


            ########
            target = torch.argmax(a, dim=1)
            pred_score = torch.argmax(pred, dim=1)
            pred_list.extend(pred_score.cpu().numpy().tolist())
            label_list.extend(target.cpu().numpy().tolist())

            ########
            for i in range(len(ans_type)):
                if ans_type[i] == 'other':
                    label_list_open.extend([target[i].cpu().numpy().tolist()])
                    pred_list_open.extend([pred_score[i].cpu().numpy().tolist()])
                elif ans_type[i] == 'yes/no':
                    label_list_close.extend([target[i].cpu().numpy().tolist()])
                    pred_list_close.extend([pred_score[i].cpu().numpy().tolist()])
            ########

            for i in range(len(ans_type)):
                if ans_type[i] == 'other':
                    open_ended += 1
                    if target[i] == pred_score[i]:
                        open_score += 1
                elif ans_type[i] == 'yes/no':
                    closed_ended += 1
                    if target[i] == pred_score[i]:
                        close_score += 1
                elif ans_type[i] == 'number':
                    number_ended += 1
                    if target[i] == pred_score[i]:
                        number_score += 1

        score += open_score + close_score + number_score


    score = (score / total)
    open_score = (open_score / open_ended)
    close_score = (close_score / closed_ended)
    number_score = (number_score / number_ended)

    return score, open_score, close_score, number_score








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
    dictionary = dataset_PathVQA.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
    train_set = dataset_PathVQA.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
    ans_token_dict = train_set.label_token_dict



    batch_size = args.batch_size

    model = Net(args, len(train_set.label2ans), len(dictionary), ans_token_dict)
    model.cuda()


    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001, [0.9, 0.999])

    score_acc, open_score_acc, close_score_acc, _ = new_evaluate(model, args)
    print(score_acc)
    print(open_score_acc)
    print(close_score_acc)
