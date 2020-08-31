from seada.butd import baseline_model as model
import os.path
import operator
import copy
import json
import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
import config
from seada import data
from seada import utils

val_loader = data.get_loader(val=True, sea=True)
logs = torch.load('logs/bs256.pth')
question_keys = logs['vocab']['question'].keys()
model = model.Net(question_keys)
model = nn.DataParallel(model).cuda()
model.module.load_state_dict(logs['weights'])

model.eval()


def sort_sample(order, *args):
    var_params = {
        'requires_grad': False,
    }
    args = [[arg[q_len_a[1]] for q_len_a in order] for arg in args]
    args = [Variable(torch.stack(arg, dim=0).cuda(), **var_params) for arg in args]
    return args

flips = 0
total = 0
tracker = utils.Tracker()
loader = tqdm(val_loader, desc='{}'.format('val'), ncols=0)
tracker_class, tracker_params = tracker.MeanMonitor, {}
perturbed_acc_tracker = tracker.track('{}_advacc'.format('val'), tracker_class(**tracker_params))
acc_tracker = tracker.track('{}_acc'.format('val'), tracker_class(**tracker_params))
for v, q, q_adv, q_str, a, b, idx, v_mask, q_mask, q_mask_adv, image_id, q_id, q_len_adv, q_len in loader:
    var_params = {
        'requires_grad': False,
    }
    v = Variable(v.cuda(), **var_params)
    q = Variable(q.cuda(), **var_params)
    a = Variable(a.cuda(), **var_params)
    b = Variable(b.cuda(), **var_params)
    q_len = Variable(q_len.cuda(), **var_params)
    v_mask = Variable(v_mask.cuda(), **var_params)
    q_mask = Variable(q_mask.cuda(), **var_params)
    answer = utils.process_answer(a)

    clean_out = model(v, b, q, v_mask, q_mask, q_len)
    clean_loss = utils.calculate_loss(answer, clean_out, method=config.loss_method)
    clean_acc, _ = utils.batch_accuracy(clean_out, answer)
    acc_tracker.append(clean_acc.mean())

    q_lens = [(q_len_adv[i], i) for i in range(q_len_adv.shape[0])]
    q_lens = sorted(q_lens, key=lambda x: x[0], reverse=True)
    q_len_adv = [q_len_a[0] for q_len_a in q_lens]
    q_len_adv = Variable(torch.stack(q_len_adv, dim=0).cuda(), **var_params)
    v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, answer, clean_out_sorted = sort_sample(q_lens, v, b,
                                                                                    q_adv, v_mask,
                                                                                    q_mask_adv,
                                                                                    answer, clean_out)

    perturbed_out = model(v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
    clean_logits = torch.max(clean_out_sorted, 1)[1].cpu().numpy()
    perturbed_logits = torch.max(perturbed_out, 1)[1].cpu().numpy()
    flips += sum(clean_logits != perturbed_logits)
    total += 256
    flip_rate = flips / total

    perturbed_loss = utils.calculate_loss(answer, perturbed_out, method=config.loss_method)
    perturbed_acc, _ = utils.batch_accuracy(perturbed_out, answer)

    perturbed_acc_tracker.append(perturbed_acc.mean())
    fmt = '{:.4f}'.format
    loader.set_postfix(flip=fmt(flip_rate), advacc=fmt(perturbed_acc_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
