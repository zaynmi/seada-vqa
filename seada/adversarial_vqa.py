
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
from . import data
from .attacks import FGSMAttack, IFGSMAttack, RandomNoise, SEA
if config.model_type == 'baseline':
    from .butd import baseline_model as model
from . import utils


class AdversarialAttackVQA:
    def __init__(self, args):
        self.args = args
        if args.name:
            self.name = ' '.join(args.name)
        else:
            self.name = '%s_%s_%s_%s_e%s_it%d_a%s_w%s_ad%s_ld%s_ade%s_fr%s' % \
                        (config.model_type, args.advtrain_data, args.attack_al, args.attack_mode, args.epsilon, args.iteration, args.alpha,
                         args.advloss_w, args.adv_delay, args.lr_decay, args.adv_end, args.samples_frac)
        self.target_name = os.path.join('logs', '{}.pth'.format(self.name))
        self.src = open(os.path.join('seada/butd', config.model_type + '_model.py')).read()
        self.config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

        self.attack_al = args.attack_al.split(',')

        self.attack_dict = {'fgsm': FGSMAttack(args.epsilon),
                       'ifgsm': IFGSMAttack(args.epsilon, args.iteration, args.alpha, False),
                       'pgd': IFGSMAttack(args.epsilon, args.iteration, args.alpha, True),
                       'noise': RandomNoise(args.epsilon),
                       'sea': SEA(fliprate=args.fliprate, topk=args.topk) if (not args.advtrain) and 'sea' in self.attack_al and len(self.attack_al) == 1 else None}
        if len(self.attack_al) == 1:
            self.adversarial = self.attack_dict[self.attack_al[0]]
        else:
            if self.attack_al[1] == 'sea':
                self.adversarial = self.attack_dict[self.attack_al[0]]
            else:
                pass

        #### generate adversarial example setting ####
        if args.generate_adv_example:
            if not args.attacked_checkpoint:
                raise ValueError('checkpoint must be provided when generate adversarial examples')
            logs = torch.load(args.attacked_checkpoint)
            self.question_keys = logs['vocab']['question'].keys()
            self.base_model = model.Net(self.question_keys)
            self.base_model = nn.DataParallel(self.base_model).cuda()

            self.base_model.module.load_state_dict(logs['weights'])
            if args.attack_only:
                if self.attack_dict['sea'] is None and 'sea' in self.attack_al:
                    self.val_loader = data.get_loader(val=True, sea=True)
                elif 'sea' in self.attack_al:
                    self.val_loader = data.get_loader(train=True, vqacp=args.vqacp)
                    self.adversarial.dataset = self.val_loader.dataset
                    self.questions_adv_saver = []
                else:
                    self.val_loader = data.get_loader(val=True)
            for param in self.base_model.parameters():
                param.requires_grad = False
            # if not args.advtrain:
            #     self.adversarial.model = self.base_model
            # if args.attack_al == 'fgsm':
            #     self.adversary = FGSMAttack(config.epsilon, self.base_model)
            # elif args.attack_al == 'ifgsm':
            #     self.adversary = IFGSMAttack(config.epsilon, config.ifgsm_iteration, config.alpha, False, self.base_model)
            # elif args.attack_al == 'pgd':
            #     self.adversary = IFGSMAttack(config.epsilon, config.ifgsm_iteration, config.alpha, True,
            #                                  self.base_model)
            # self.fgsm = FGSMAttack(config.epsilon, self.base_model)
            # self.ifgsm = IFGSMAttack(config.epsilon, config.ifgsm_iteration, config.alpha, False, self.base_model)
            # self.tfgsm = TargetedFGSM(config.epsilon, self.base_model)
        if args.advtrain:
            print('will save to {}'.format(self.target_name))
            cudnn.benchmark = True
            if args.resume:
                logs = torch.load(args.resume)
                # hacky way to tell the VQA classes that they should use the vocab without passing more params around
                data.preloaded_vocab = logs['vocab']
            if args.advtrain_data == 'trainval':
                if 'sea' in self.attack_al:
                    self.train_loader = data.get_loader(trainval=True, sea=True, vqacp=self.args.vqacp)
                else:
                    self.train_loader = data.get_loader(trainval=True, vqacp=self.args.vqacp)
            else:
                if 'sea' in self.attack_al:
                    self.train_loader = data.get_loader(train=True, sea=True, frac=args.samples_frac, vqacp=self.args.vqacp)
                    self.val_loader = data.get_loader(val=True, sea=True, vqacp=self.args.vqacp)
                else:
                    self.train_loader = data.get_loader(train=True, frac=args.samples_frac, vqacp=self.args.vqacp)
                    self.val_loader = data.get_loader(val=True, vqacp=self.args.vqacp)
            if self.attack_dict['sea'] is not None:
                self.adversarial.dataset = self.train_loader.dataset
            self.question_keys = self.train_loader.dataset.vocab['question'].keys() if args.advtrain_data == 'trainval' else \
            self.val_loader.dataset.vocab[
                'question'].keys()
            self.model = model.Net(self.question_keys)
            self.model = nn.DataParallel(self.model).cuda()
            # if args.resume:
            #     print('loading weights from %s' % args.resume)
            #     self.model.module.load_state_dict(logs['weights'])
            self.start_epoch = 0
            self.select_optim = optim.Adamax if (config.optim_method == 'Adamax') else optim.Adam
            self.optimizer = self.select_optim([p for p in self.model.parameters() if p.requires_grad], lr=config.initial_lr,
                                     weight_decay=config.weight_decay)
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, 0.5 ** (1 / config.lr_halflife))
            if args.resume:
                print('loading weights from %s' % args.resume)
                if config.model_type == 'counting':
                    self.model.load_state_dict(logs['weights'])
                else:
                    self.model.module.load_state_dict(logs['weights'])
                self.optimizer.load_state_dict(logs['optimizer'])
                if config.model_type == 'counting':
                    self.scheduler.load_state_dict(logs['scheduler'])
                self.start_epoch = logs['epoch']

        if args.eval_advtrain or args.test_advtrain:
            if args.checkpoint:
                logs = torch.load(args.checkpoint)
                # hacky way to tell the VQA classes that they should use the vocab without passing more params around
                data.preloaded_vocab = logs['vocab']
            self.val_loader = data.get_loader(val=True, sea=True if 'sea' in self.attack_al else False) if args.eval_advtrain else data.get_loader(test=True)
            self.question_keys = self.val_loader.dataset.vocab['question'].keys()
            self.model = model.Net(self.question_keys)
            self.model = nn.DataParallel(self.model).cuda()
            if args.checkpoint:
                print('loading weights from %s' % args.checkpoint)
                self.model.module.load_state_dict(logs['weights'])

        self.tracker = utils.Tracker()

    def attack(self, loader):
        tracker_class, tracker_params = self.tracker.MeanMonitor, {}
        loader = tqdm(loader, desc='{} '.format(self.args.attack_al), ncols=0)
        loss_tracker = self.tracker.track('{}_loss'.format('attack'), tracker_class(**tracker_params))
        acc_tracker = self.tracker.track('{}_acc'.format('before attack'), tracker_class(**tracker_params))
        perturbed_acc_tracker = self.tracker.track('{}_acc'.format('after attack'), tracker_class(**tracker_params))
        dist_tracker = self.tracker.track('{}_dist'.format('dist'), tracker_class(**tracker_params))
        if len(self.attack_al) == 2:
            vqc_q_tracker = self.tracker.track('{}_acc'.format('after attack'), tracker_class(**tracker_params))
            vqadv_q_tracker = self.tracker.track('{}_acc'.format('after attack'), tracker_class(**tracker_params))
            vqc_qadv_tracker = self.tracker.track('{}_acc'.format('after attack'), tracker_class(**tracker_params))
            vqadv_qadv_tracker = self.tracker.track('{}_acc'.format('after attack'), tracker_class(**tracker_params))
        self.adversarial.model = self.base_model
        for v, q, q_adv, q_str, a, b, idx, v_mask, q_mask, q_mask_adv, image_id, q_id, q_len_adv, q_len in loader:
            var_params = {
                'requires_grad': False,
            }
            v = v.cuda()
            q = Variable(q.cuda())
            a = Variable(a.cuda())
            b = Variable(b.cuda())
            q_len = Variable(q_len.cuda())
            v_mask = Variable(v_mask.cuda())
            q_mask = Variable(q_mask.cuda())
            answer = utils.process_answer(a)

            if self.args.attack_mode == 'q':
                if self.attack_al[0] == 'sea':
                    clean_out = self.base_model(v, b, q, v_mask, q_mask, q_len)
                    clean_logits = torch.max(clean_out, 1)[1].cpu().numpy()
                    v, b, v_mask, q_adv, q_len_adv, q_mask_adv, answer, q_str_adv, image_id_adv, q_id_adv = self.adversarial.perturb((v, b, q, q_str, v_mask, q_mask, image_id, q_id, q_len), y=answer, oripred=clean_logits)
                    perturbed_out = self.base_model(v, b, q_adv, v_mask, q_mask_adv, q_len_adv)
                    self.save_q_adv(q_str_adv, image_id_adv, q_id_adv)
                    dist = 0
                    dist_tracker.append(dist)
                else:
                    q_adv = self.adversarial.perturb((v, b, q, v_mask, q_mask, q_len), answer, perturb_q=True)
                    perturbed_out = self.base_model(v, b, q, v_mask, q_mask, q_len, q_adv)
                    dist = self.distance(q_adv, self.base_model.module.text.embedded.detach())
                    dist_tracker.append(dist.data)
            elif self.args.attack_mode == 'v':
                v_adv, acc, loss = self.adversarial.perturb((v, b, q, v_mask, q_mask, q_len), answer)
                perturbed_out = self.base_model(v_adv, b, q, v_mask, q_mask, q_len)
                dist = self.distance(v, v_adv)
                dist_tracker.append(dist.data)

            elif self.args.attack_mode == 'vq':    # todo: v cooperate q
                q_lens = [(q_len_adv[i], i) for i in range(q_len_adv.shape[0])]
                q_lens = sorted(q_lens, key=lambda x: x[0], reverse=True)
                q_len_adv = [q_len_a[0] for q_len_a in q_lens]
                q_len_adv = Variable(torch.stack(q_len_adv, dim=0).cuda(), **var_params)
                v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, answer_sorted = self.sort_sample(q_lens, v, b,
                                                                                                       q_adv, v_mask,
                                                                                                       q_mask_adv,
                                                                                                       answer)

                v_qadv, _, _ = self.adversarial.perturb((v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv), answer_sorted)
                v_qc, _, _ = self.adversarial.perturb((v, b, q, v_mask, q_mask, q_len), answer)
                v_qadv = Variable(v_qadv.data, requires_grad=False)
                v_qc = Variable(v_qc.data, requires_grad=False)

                q_sorted, q_mask_sorted, q_len_sorted, v_qc_sorted = self.sort_sample(q_lens, q, q_mask, q_len, v_qc)
                out_vqadv_qadv = self.base_model(v_qadv, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
                vqadv_qadv_acc, _ = utils.batch_accuracy(out_vqadv_qadv, answer_sorted)
                vqadv_qadv_tracker.append(vqadv_qadv_acc.data.cpu().mean())

                out_vqc_q = self.base_model(v_qc, b, q, v_mask, q_mask, q_len)
                vqc_q_acc, _ = utils.batch_accuracy(out_vqc_q, answer)
                vqc_q_tracker.append(vqc_q_acc.data.cpu().mean())

                v_qadv_re = self.restore_order(q_lens, v_qadv)[0]
                out_vqadv_q = self.base_model(v_qadv_re, b, q, v_mask, q_mask, q_len)
                vqadv_q_acc, _ = utils.batch_accuracy(out_vqadv_q, answer)
                vqadv_q_tracker.append(vqadv_q_acc.data.cpu().mean())

                out_vqc_qadv = self.base_model(v_qc_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
                vqc_qadv_acc, _ = utils.batch_accuracy(out_vqc_qadv, answer_sorted)
                vqc_qadv_tracker.append(vqc_qadv_acc.data.cpu().mean())

                fmt = '{:.4f}'.format
                loader.set_postfix(vqc_q_acc=fmt(vqc_q_tracker.mean.value),
                                   vqadv_q_acc=fmt(vqadv_q_tracker.mean.value),
                                   vqc_qadv_acc=fmt(vqc_qadv_tracker.mean.value),
                                   vqadv_qadv_acc=fmt(vqadv_qadv_tracker.mean.value))
                continue
            else:
                perturbed_out = self.base_model(v, b, q, v_mask, q_mask, q_len)
                dist = 0
                dist_tracker.append(dist)

            perturbed_acc, _ = utils.batch_accuracy(perturbed_out, answer)
            loss = utils.calculate_loss(answer, perturbed_out, method=config.loss_method)

            loss_tracker.append(loss.item())
            # acc_tracker.append(acc.mean())

            perturbed_acc_tracker.append(perturbed_acc.data.cpu().mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value),# acc=fmt(acc_tracker.mean.value),
                               acc_after_attack=fmt(perturbed_acc_tracker.mean.value),
                               distance=fmt(dist_tracker.mean.value))
        if self.args.attack_al == 'sea':
            with open('data/vqacp/vqacp_v2_train_questions_adv.json', 'w') as f:
                json.dump({'questions': self.questions_adv_saver}, f)
        if len(self.attack_al) == 1:
            f = open('attack_log.txt', 'a')
            f.write(self.name + '\n')
            f.write(str(fmt(perturbed_acc_tracker.mean.value)))
            f.write('\n')
            f.write(str(fmt(dist_tracker.mean.value)))
            f.write('\n')
        else:
            f = open('attack_log.txt', 'a')
            f.write(self.name + '\n')
            f.write('vqc_q: ' + str(fmt(vqc_q_tracker.mean.value)) + ' vqadv_q: ' + str(fmt(vqadv_q_tracker.mean.value))
                    + ' vqc_qadv: ' + str(fmt(vqc_qadv_tracker.mean.value)) + ' vqadv_qadv: ' + str(
                fmt(vqadv_qadv_tracker.mean.value)))
            f.write('\n')

    def advsarial_training(self):
        best_valid = 0
        lr_decay_epochs = range(self.args.lr_decay, 100, 2)
        for epoch in range(self.start_epoch, config.epochs):
            self.model.train()
            tracker_class, tracker_params = self.tracker.MovingMeanMonitor, {'momentum': 0.99}
            if epoch < len(config.gradual_warmup_steps) and config.schedule_method == 'warm_up':
                utils.set_lr(self.optimizer, config.gradual_warmup_steps[epoch])
                utils.print_lr(self.optimizer, 'train', epoch)
            elif (epoch in lr_decay_epochs) and config.schedule_method == 'warm_up':
                utils.decay_lr(self.optimizer, config.lr_decay_rate)
                utils.print_lr(self.optimizer, 'train', epoch)
            else:
                utils.print_lr(self.optimizer, 'train', epoch)
            loader = tqdm(self.train_loader, desc='{} E{:03d}'.format('train', epoch), ncols=0)
            loss_tracker = self.tracker.track('{}_loss'.format('train'), tracker_class(**tracker_params))
            acc_tracker = self.tracker.track('{}_acc'.format('train'), tracker_class(**tracker_params))

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

                out = self.model(v, b, q, v_mask, q_mask, q_len)
                answer = utils.process_answer(a)
                loss = utils.calculate_loss(answer, out, method=config.loss_method)
                acc, y_pred = utils.batch_accuracy(out, answer)
                if self.args.adv_delay < epoch + 1 < self.args.adv_end:
                    # use predicted label to prevent label leaking
                    if 'sea' in self.attack_al:
                        q_lens = [(q_len_adv[i], i) for i in range(q_len_adv.shape[0])]
                        q_lens = sorted(q_lens, key=lambda x: x[0], reverse=True)
                        q_len_adv = [q_len_a[0] for q_len_a in q_lens]
                        q_len_adv = Variable(torch.stack(q_len_adv, dim=0).cuda(), **var_params)
                        v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, answer_sorted = self.sort_sample(q_lens, v, b, q_adv, v_mask, q_mask_adv, answer)

                    if self.args.attack_mode == 'q':
                        if self.attack_al[0] == 'sea':
                            out_adv = self.model(v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
                            loss_adv = [utils.calculate_loss(answer_sorted, out_adv, method=config.loss_method)]
                        else:
                            v_adv = self.advtrain_step((v, b, q, v_mask, q_mask, q_len), y_pred, self.model,
                                                       self.adversarial, True)
                            v_adv = Variable(v_adv.data, requires_grad=False)
                            out_adv = self.model(v, b, q, v_mask, q_mask, q_len, v_adv)
                            loss_adv = [utils.calculate_loss(answer, out_adv, method=config.loss_method)]

                    elif self.args.attack_mode == 'v':
                        v_adv = self.advtrain_step((v, b, q, v_mask, q_mask, q_len), y_pred, self.model,
                                                   self.adversarial, False)
                        v_adv = Variable(v_adv.data, requires_grad=False)
                        out_adv = self.model(v_adv, b, q, v_mask, q_mask, q_len)
                        loss_adv = [utils.calculate_loss(answer, out_adv, method=config.loss_method)]
                    else:
                        if self.attack_al[1] == 'sea':   # todo: v & q
                            y_pred_sorted = self.sort_sample(q_lens, y_pred)[0]
                            v_qadv = self.advtrain_step((v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv), y_pred_sorted, self.model,
                                                       self.adversarial, False)
                            v_qc = self.advtrain_step((v, b, q, v_mask, q_mask, q_len), y_pred, self.model,
                                                       self.adversarial, False)

                            v_qadv = Variable(v_qadv.data, requires_grad=False)
                            v_qc = Variable(v_qc.data, requires_grad=False)
                            q_sorted, q_mask_sorted, q_len_sorted, v_qc_sorted = self.sort_sample(q_lens, q, q_mask, q_len, v_qc)
                            out_vqadv_qadv = self.model(v_qadv, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
                            loss_vqadv_qadv = utils.calculate_loss(answer_sorted, out_vqadv_qadv, method=config.loss_method)
                            out_vqc_q = self.model(v_qc, b, q, v_mask, q_mask, q_len)
                            loss_vqc_q = utils.calculate_loss(answer, out_vqc_q, method=config.loss_method)
                            v_qadv_re = self.restore_order(q_lens, v_qadv)[0]
                            out_vqadv_q = self.model(v_qadv_re, b, q, v_mask, q_mask, q_len)
                            loss_vqadv_q = utils.calculate_loss(answer, out_vqadv_q, method=config.loss_method)
                            out_vqc_qadv = self.model(v_qc_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
                            loss_vqc_qadv = utils.calculate_loss(answer_sorted, out_vqc_qadv, method=config.loss_method)
                            loss_adv = [loss_vqadv_qadv, loss_vqc_q, loss_vqadv_q, loss_vqc_qadv]

                    loss = (self.args.advloss_w * sum(loss_adv) + loss) / (len(loss_adv) + 1)

                self.optimizer.zero_grad()
                loss.backward()
                # print gradient
                if config.print_gradient:
                    utils.print_grad([(n, p) for n, p in self.model.named_parameters() if p.grad is not None])
                # clip gradient
                clip_grad_norm_(self.model.parameters(), config.clip_value)
                self.optimizer.step()
                if (config.schedule_method == 'batch_decay'):
                    self.scheduler.step()
                loss_tracker.append(loss.item())
                acc_tracker.append(acc.data.cpu().mean())
                fmt = '{:.4f}'.format
                loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
            if self.args.advtrain_data != 'trainval':
                r = self.evaluate(self.val_loader)
                if epoch == config.advtrain_delay:
                    best_valid = 0
                if sum(r[1]) / len(r[1]) > best_valid:
                    best_valid = sum(r[1]) / len(r[1])
                    print('best valid')
                    results = {
                        'name': self.name,
                        'tracker': self.tracker.to_dict(),
                        'config': self.config_as_dict,
                        'weights': self.model.module.state_dict(),
                        'eval': {
                            'clean_answers': r[0],
                            'clean_accuracies': r[1],
                            'adv_answers': r[3],
                            'adv_accuracies': r[4],
                            'idx': r[2],
                        },
                        'vocab': self.val_loader.dataset.vocab if self.args.advtrain_data == 'train' else self.train_loader.dataset.vocab,
                        'src': self.src,
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict() if config.model_type == 'counting' else [],
                        'epoch': epoch + 1,
                    }
                    torch.save(results, self.target_name)
            else:
                r = [[-1], [-1], [-1], [-1], [-1]]
                results = {
                    'name': self.name,
                    'tracker': self.tracker.to_dict(),
                    'config': self.config_as_dict,
                    'weights': self.model.module.state_dict(),
                    'eval': {
                        'clean_answers': r[0],
                        'clean_accuracies': r[1],
                        'adv_answers': r[3],
                        'adv_accuracies': r[4],
                        'idx': r[2],
                    },
                    'vocab': self.val_loader.dataset.vocab if self.args.advtrain_data == 'train' else self.train_loader.dataset.vocab,
                    'src': self.src,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict() if config.model_type == 'counting' else [],
                    'epoch': epoch + 1,
                }
                torch.save(results, self.target_name)

        f = open('log.txt', 'a')
        f.write(self.name + '\n')
        f.write(str(best_valid.data.cpu().numpy()))
        f.write('\n')
        f.write(str((sum(results['eval']['adv_accuracies'])/len(results['eval']['adv_accuracies'])).data.cpu()))
        f.write('\n')

    def advtrain_step(self, X, y, net, adversary, perturb_q):
        # If adversarial training, need a snapshot of
        # the model at each batch to compute grad, so
        # as not to mess up with the optimization step
        # model_cp = copy.deepcopy(net)
        model_cp = model.Net(self.question_keys)
        model_cp = nn.DataParallel(model_cp).cuda()
        model_cp.load_state_dict(net.state_dict())
        for p in model_cp.parameters():
            p.requires_grad = False
        # model_cp.eval()

        adversary.model = model_cp

        if perturb_q:
            X_adv = adversary.perturb(X, y, perturb_q=True)
        else:
            X_adv, _, _ = adversary.perturb(X, y)

        return X_adv

    def sort_sample(self, order, *args):
        var_params = {
            'requires_grad': False,
        }
        args = [[arg[q_len_a[1]] for q_len_a in order] for arg in args]
        args = [Variable(torch.stack(arg, dim=0).cuda(), **var_params) for arg in args]
        return args

    def restore_order(self, order, *args):
        var_params = {
            'requires_grad': False,
        }
        args = [[(arg[i], q_len_a[1]) for i, q_len_a in enumerate(order)] for arg in args]
        args = [sorted(arg, key=lambda x: x[1]) for arg in args]
        args = [[ar[0] for ar in arg] for arg in args]
        args = [Variable(torch.stack(arg, dim=0).cuda(), **var_params) for arg in args]
        return args

    def vq_loss(self, loss, loss_adv):
        return (sum(loss_adv) + loss) / (len(loss_adv) + 1)

    def evaluate(self, loader, has_answers=True):
        self.model.eval()
        tracker_class, tracker_params = self.tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []
        perturbed_answ = []
        perturbed_accs = []
        if self.args.attacked_checkpoint and self.attack_dict['sea'] is None:
            self.adversarial.model = self.base_model
        loader = tqdm(loader, desc='{}'.format('val'), ncols=0)
        loss_tracker = self.tracker.track('{}_loss'.format('val'), tracker_class(**tracker_params))
        acc_tracker = self.tracker.track('{}_acc'.format('val'), tracker_class(**tracker_params))
        perturbed_loss_tracker = self.tracker.track('{}_advloss'.format('val'), tracker_class(**tracker_params))
        perturbed_acc_tracker = self.tracker.track('{}_advacc'.format('val'), tracker_class(**tracker_params))

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

            clean_out = self.model(v, b, q, v_mask, q_mask, q_len)
            if has_answers:
                answer = utils.process_answer(a)    # answer must be known when generate adversarial example
                clean_loss = utils.calculate_loss(answer, clean_out, method=config.loss_method)
                clean_acc, _ = utils.batch_accuracy(clean_out, answer)
                accs.append(clean_acc.data.cpu().view(-1))
                loss_tracker.append(clean_loss.item())
                acc_tracker.append(clean_acc.mean())

                if self.args.attacked_checkpoint:
                    # if self.args.attack_al == 'fgsm':
                    #     v_adv, acc, loss = self.fgsm.perturb((v, b, q, v_mask, q_mask, q_len), answer)
                    # elif self.args.attack_al == 'ifgsm':
                    #     v_adv, acc, loss = self.ifgsm.perturb((v, b, q, v_mask, q_mask, q_len), answer)
                    if 'sea' in self.attack_al:
                        q_lens = [(q_len_adv[i], i) for i in range(q_len_adv.shape[0])]
                        q_lens = sorted(q_lens, key=lambda x: x[0], reverse=True)
                        q_len_adv = [q_len_a[0] for q_len_a in q_lens]
                        q_len_adv = Variable(torch.stack(q_len_adv, dim=0).cuda(), **var_params)
                        v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, answer = self.sort_sample(q_lens, v, b,
                                                                                                        q_adv, v_mask,
                                                                                                        q_mask_adv,
                                                                                                        answer)
                    if self.args.attack_mode == 'q':
                        if self.attack_al[0] == 'sea':
                            perturbed_out = self.model(v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)
                        else:
                            q_adv = self.adversarial.perturb((v, b, q, v_mask, q_mask, q_len), answer, perturb_q=True)
                            q_adv = Variable(q_adv.data, requires_grad=False)
                            perturbed_out = self.model(v, b, q, v_mask, q_mask, q_len, q_adv)
                    elif self.args.attack_mode == 'v':
                        v_adv, acc, loss = self.adversarial.perturb((v, b, q, v_mask, q_mask, q_len), answer)
                        perturbed_out = self.model(v_adv, b, q, v_mask, q_mask, q_len)
                    else:  # todo: eval v & q
                        if 'sea' in self.attack_al:
                            v_adv, acc, loss = self.adversarial.perturb((v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv), answer)
                            perturbed_out = self.model(v_sorted, b_sorted, q_adv, v_mask_sorted, q_mask_adv, q_len_adv)

                    perturbed_loss = utils.calculate_loss(answer, perturbed_out, method=config.loss_method)
                    perturbed_acc, _ = utils.batch_accuracy(perturbed_out, answer)
                    _, perturbed_answer = perturbed_out.data.cpu().max(dim=1)
                    perturbed_answ.append(perturbed_answer.view(-1))
                    perturbed_accs.append(perturbed_acc.data.cpu().view(-1))
                    perturbed_loss_tracker.append(perturbed_loss.item())
                    perturbed_acc_tracker.append(perturbed_acc.mean())
                    fmt = '{:.4f}'.format
                    loader.set_postfix(advloss=fmt(perturbed_loss_tracker.mean.value),
                                       advacc=fmt(perturbed_acc_tracker.mean.value),
                                       loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))
                else:
                    fmt = '{:.4f}'.format
                    loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

            _, clean_answer = clean_out.data.cpu().max(dim=1)
            answ.append(clean_answer.view(-1))
            idxs.append(idx.view(-1).clone())

        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
            if self.args.attacked_checkpoint:
                perturbed_accs = list(torch.cat(perturbed_accs, dim=0))
                perturbed_answ = list(torch.cat(perturbed_answ, dim=0))
        idxs = list(torch.cat(idxs, dim=0))

        return answ, accs, idxs, perturbed_answ, perturbed_accs

    def save_result_json(self, loader, has_answers=True):
        r = self.evaluate(loader, has_answers)
        answer_index_to_string = {a: s for s, a in loader.dataset.answer_to_index.items()}
        results = []
        for answer, index in zip(r[0], r[2]):
            answer = answer_index_to_string[answer.item()]
            qid = loader.dataset.question_ids[index]
            entry = {
                'question_id': qid,
                'answer': answer,
            }
            results.append(entry)
        with open(config.result_json_path, 'w') as fd:
            json.dump(results, fd)

    def load_checkpoint(self, path):
        logs = torch.load(' '.join(path))
        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        data.preloaded_vocab = logs['vocab']
        self.model.module.load_state_dict(logs['weights'])

    def distance(self, x, x_adv):
        dist = torch.norm(x - x_adv, 2, 2) / x.shape[2] ** 0.5
        return torch.mean(dist)

    def save_q_adv(self, q_str, image_id, q_id):
        assert len(q_str) == len(image_id) == len(q_id)
        for i in range(len(q_str)):
            f = {'image_id': int(image_id[i]), 'question': q_str[i], 'question_id': int(q_id[i])}
            self.questions_adv_saver.append(f)
