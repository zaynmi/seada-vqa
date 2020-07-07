import torch
import utils
import config
from torch.autograd import Variable
import data
import numpy as np
import spacy

from sea.paraphrase_scorer import ParaphraseScorer
from sea import onmt_model, replace_rules

# --- White-box attacks ---
inter_feature = {}
inter_gradient = {}


def make_hook(name, flag):
    if flag == 'forward':
        def hook(m, input, output):
            inter_feature[name] = input
        return hook
    elif flag == 'backward':
        def hook(m, input, output):
            inter_gradient[name] = output
        return hook
    else:
        assert False


class FGSMAttack(object):
    def __init__(self, epsilon=None, model=None):
        """
        One step fast gradient sign method
        """
        self.model = model
        self.epsilon = epsilon

    def perturb(self, X_nat, y, epsilon=None, k=None, alpha=None, perturb_q=False, targeted=False):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        v, b, q, v_mask, q_mask, q_len = X_nat
        v = Variable(v, requires_grad=True)
        if not perturb_q:
            # Providing epsilons in batch
            if epsilon is not None:
                self.epsilon = epsilon
            out = self.model(v, b, q, v_mask, q_mask, q_len)

            loss = utils.calculate_loss(y, out, method=config.loss_method)
            acc, _ = utils.batch_accuracy(out, y)
            self.model.zero_grad()
            loss.backward()

            if targeted:
                data_grad = -v.grad.data.sign()
            else:
                data_grad = v.grad.data.sign()
            perturbed_v = v + self.epsilon * data_grad

            return perturbed_v, acc.data.cpu(), loss
        else:
            out = self.model(v, b, q, v_mask, q_mask, q_len)
            loss = utils.calculate_loss(y, out, method=config.loss_method)
            # self.model.module.text.embed.register_backward_hook(make_hook('int_emb', 'backward'))
            self.model.zero_grad()
            loss.backward()
            if targeted:
                data_grad = -self.model.module.text.embedded.grad.data.sign()
            else:
                data_grad = self.model.module.text.embedded.grad.data.sign()
            perturbed_q = self.model.module.text.embedded.detach()
            perturbed_q = perturbed_q + config.epsilon * data_grad
            return perturbed_q


class IFGSMAttack(object):
    def __init__(self, epsilon=0.3, k=40, alpha=0.01,
        random_start=True, model=None):
        """
        Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point.
        https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
        """
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha
        self.rand = random_start  # if true, it is PGD attack

    def perturb(self, X_nat, y, epsilon=None, k=None, alpha=None, perturb_q=False, targeted=False):
        """
        Given examples (X_nat, y), returns adversarial
        examples within epsilon of X_nat in l_infinity norm.
        """
        v_nat, b, q_nat, v_mask, q_mask, q_len = X_nat
        if epsilon is not None:
            self.epsilon = epsilon
        if k is not None:
            self.k = k
        if alpha is not None:
            self.alpha = alpha
        if not perturb_q:
            if self.rand:
                v = v_nat + torch.FloatTensor(v_nat.size()).uniform_(-self.epsilon, self.epsilon).cuda()
            else:
                v = v_nat.clone()
            v_adv = Variable(v, requires_grad=True)

            for i in range(self.k):
                out = self.model(v_adv, b, q_nat, v_mask, q_mask, q_len)
                loss = utils.calculate_loss(y, out, method=config.loss_method)
                acc, pred_out = utils.batch_accuracy(out, y)
                self.model.zero_grad()
                loss.backward()
                if targeted:
                    data_grad = -v_adv.grad.data.sign()
                else:
                    data_grad = v_adv.grad.data.sign()
                v_adv = v_adv + self.alpha * data_grad
                v_adv = utils.where(v_adv > v_nat + self.epsilon, v_nat + self.epsilon, v_adv)
                v_adv = utils.where(v_adv < v_nat - self.epsilon, v_nat - self.epsilon, v_adv)
                # v_adv = torch.clamp(v_adv, v - config.epsilon, v + config.epsilon)
                v_adv = Variable(v_adv.data, requires_grad=True)

            return v_adv, acc.data.cpu(), loss
        else:
            out = self.model(v_nat, b, q_nat, v_mask, q_mask, q_len)
            loss = utils.calculate_loss(y, out, method=config.loss_method)
            # self.model.module.text.embed.register_backward_hook(make_hook('int_emb', 'backward'))
            self.model.zero_grad()
            loss.backward()
            data_grad = self.model.module.text.embedded.grad.data.sign()
            origin_q = self.model.module.text.embedded.detach()
            perturbed_q = origin_q + self.alpha * data_grad
            for i in range(1, self.k):
                out = self.model(v_nat, b, q_nat, v_mask, q_mask, q_len, perturbed_q)
                loss = utils.calculate_loss(y, out, method=config.loss_method)
                # acc, pred_out = utils.batch_accuracy(out, y)
                self.model.zero_grad()
                loss.backward()
                if targeted:
                    data_grad = -self.model.module.text.embedded.grad.data.sign()
                else:
                    data_grad = self.model.module.text.embedded.grad.data.sign()
                perturbed_q = self.model.module.text.embedded.detach() + self.alpha * data_grad
                perturbed_q = utils.where(perturbed_q > origin_q + self.epsilon, origin_q + self.epsilon, perturbed_q)
                perturbed_q = utils.where(perturbed_q < origin_q - self.epsilon, origin_q - self.epsilon, perturbed_q)
                # v_adv = torch.clamp(v_adv, v - config.epsilon, v + config.epsilon)
                perturbed_q = Variable(perturbed_q.data)

            return perturbed_q

class RandomNoise(object):
    def __init__(self, epsilon, model=None):
        self.epsilon = epsilon
        self.model = model

    def perturb(self, X_nat, y, epsilon=None, k=None, alpha=None, perturb_q=False, targeted=False):
        v, b, q, v_mask, q_mask, q_len = X_nat

        v_adv = v + self.epsilon * torch.randn_like(v).cuda()
        out = self.model(v_adv, b, q, v_mask, q_mask, q_len)
        loss = utils.calculate_loss(y, out, method=config.loss_method)
        acc, _ = utils.batch_accuracy(out, y)

        return v_adv, acc.data.cpu(), loss


class SEA(object):
    def __init__(self, dataset=None, model=None, fliprate=0, topk=None):
        self.dataset = dataset
        self.model = model
        self.ps = ParaphraseScorer(gpu_id=0)
        self.nlp = spacy.load('en')
        self.fliprate = fliprate
        #self.ratetemp = fliprate
        self.topk = topk

    def perturb(self, X_nat, y=None, oripred=None, epsilon=None, k=None, alpha=None, perturb_q=False, targeted=False):
        v, b, q, q_str, v_mask, q_mask, image_id, q_id, q_len = X_nat
        q_advs = []
        q_len_clue = []
        q_mask_advs = []
        q_str_advs = []
        flips = int(v.shape[0] * self.fliprate)
        topk = self.topk
        fliprate = self.fliprate
        nflip = 0
        for i in range(v.shape[0]):
            q_adv, q_len_adv, q_mask_adv, q_str_adv, flipsign = self.find_flips(q_str[i], visual=(v[i], b[i], v_mask[i]), topk=topk, fliprate=fliprate, threshold=-10, oripred=oripred[i])
            if q_adv is None:     # support top1 right now todo: support topk
                q_adv = q[i].unsqueeze(0)
                q_len_adv = q_len[i].unsqueeze(0)
                q_mask_adv = q_mask[i].unsqueeze(0)
                q_str_adv = q_str[i]
            if flipsign:
                nflip += 1
            if nflip > flips:
                fliprate = 0
                topk = 1
            q_advs.append(q_adv)
            q_len_clue.append((q_len_adv, i))
            q_mask_advs.append(q_mask_adv)
            q_str_advs.append(q_str_adv)
        q_len_clue = sorted(q_len_clue, key=lambda x: x[0], reverse=True)
        q_len_advs = [clue[0] for clue in q_len_clue]
        sort_id = [clue[1] for clue in q_len_clue]

        q_advs = [q_advs[idx] for idx in sort_id]
        q_mask_advs = [q_mask_advs[idx] for idx in sort_id]
        q_str_advs = [q_str_advs[idx] for idx in sort_id]
        v = [v[idx] for idx in sort_id]
        b = [b[idx] for idx in sort_id]
        v_mask = [v_mask[idx] for idx in sort_id]
        y = [y[idx] for idx in sort_id]
        image_id = [image_id[idx] for idx in sort_id]
        q_id = [q_id[idx] for idx in sort_id]

        q_adv = torch.cat(q_advs, dim=0)
        q_len_adv = torch.cat(q_len_advs, dim=0)
        q_mask_adv = torch.cat(q_mask_advs, dim=0)
        v = torch.stack(v, dim=0)
        b = torch.stack(b, dim=0)
        v_mask = torch.stack(v_mask, dim=0)
        y = torch.stack(y, dim=0)
        return v, b, v_mask, q_adv, q_len_adv, q_mask_adv, y, q_str_advs, image_id, q_id

    def find_flips(self, instance, visual=None, topk=1, fliprate=0, threshold=-10, oripred=None):
        instance_for_onmt = onmt_model.clean_text(' '.join([x.text for x in self.nlp.tokenizer(instance)]), only_upper=False)
        paraphrases = self.ps.generate_paraphrases(instance_for_onmt, topk=topk+1, edit_distance_cutoff=4, threshold=threshold)
        if len(paraphrases) == 0:
            return None, None, None, None, False
        for para in paraphrases:
            if para[0] == instance_for_onmt:
                paraphrases.remove(para)
        if len(paraphrases) == 0:
            return None, None, None, None, False
        paraphrases = paraphrases[:topk]
        prepared_paraphrases = data.prepare_questions_from_para(paraphrases)
        questions = [self.dataset.encode_question(paraphrase) for paraphrase in prepared_paraphrases]
        questions = [(q_tuple[0], q_tuple[1], i) for i, q_tuple in enumerate(questions)]
        sorted_questions = sorted(questions, key=lambda x: x[1], reverse=True)
        q_len = torch.cat([torch.tensor([q[1]]) for q in sorted_questions], 0).cuda()
        q = torch.stack([q[0] for q in sorted_questions], 0).cuda()
        q_m = [torch.from_numpy((np.arange(self.dataset.max_question_length) < q[1]).astype(int)) for q in
               sorted_questions]
        q_mask = torch.stack(q_m, 0).float().cuda()
        if fliprate == 0:
            return q, q_len, q_mask, paraphrases[0][0], False
        else:
            v, b, v_mask = visual
            v = v.unsqueeze(0).repeat(q.shape[0], 1, 1)
            b = b.unsqueeze(0).repeat(q.shape[0], 1, 1)
            v_mask = v_mask.unsqueeze(0).repeat(q.shape[0], 1, 1)
            # # discard the flip or not, compute adv for every example
            perturbed_out = self.model(v, b, q, v_mask, q_mask, q_len)
            perturbed_logits = torch.max(perturbed_out, 1)[1].cpu().numpy()
            p = np.where(perturbed_logits != oripred)[0].tolist()
            sorted_para = [paraphrases[qs[2]] for qs in sorted_questions]
            flipsign = False
            if len(p) == 0:
                return q[0].unsqueeze(0), q_len[0].unsqueeze(0), q_mask[0].unsqueeze(0), sorted_para[0][0], flipsign
            else:
                flipsign = True
                return q[p[0]].unsqueeze(0), q_len[p[0]].unsqueeze(0), q_mask[p[0]].unsqueeze(0), sorted_para[p[0]][0], flipsign

        # perturbed_acc, _ = utils.batch_accuracy(perturbed_out, orig_pred.unsqueeze(0).repeat(q.shape[0], 1))



