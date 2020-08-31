import argparse
import config
import torch.backends.cudnn as cudnn
import torch
from seada.adversarial_vqa import AdversarialAttackVQA
import warnings

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--attack_only', action='store_true')
    parser.add_argument('--generate_adv_example', action='store_true')
    parser.add_argument('--attacked_checkpoint', type=str, help='must be announced when attack only')
    parser.add_argument('--attack_al', type=str, default='ifgsm', help='attack algorithm')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--attack_mode', default='v', choices=['v', 'q', 'vq', 'no'])
    parser.add_argument('--advtrain', action='store_true')
    parser.add_argument('--vqacp', action='store_true')
    parser.add_argument('--advtrain_data', default='train', choices=['train', 'trainval'])
    parser.add_argument('--eval_advtrain', action='store_true')
    parser.add_argument('--test_advtrain', action='store_true')
    parser.add_argument('--advloss_w', type=int, default=1)
    parser.add_argument('--samples_frac', type=float, default=1)
    parser.add_argument('--adv_delay', type=int, default=10)
    parser.add_argument('--adv_end', type=int, default=15)
    parser.add_argument('--epsilon', type=float, default=0.3)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--iteration', type=int, default=2)
    parser.add_argument('--lr_decay', type=int, default=15)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--fliprate', type=float, default=0)
    parser.add_argument('--paraphrase_data', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--describe', type=str, default='describe your setting')
    args = parser.parse_args()
    if args.attack_only:
        args.generate_adv_example = True

    if args.eval_advtrain or args.advtrain:
        if args.attacked_checkpoint:
            args.generate_adv_example = True

    if args.test_advtrain:
        args.attacked_checkpoint = False
        args.generate_adv_example = False

    print('-' * 50)
    print(args)
    config.print_param()

    # set mannual seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # ----------Tasks-------------------
    attackvqa = AdversarialAttackVQA(args)
    if args.attack_only:
        attackvqa.attack(attackvqa.val_loader)
    if args.advtrain:
        attackvqa.advsarial_training()

    if args.eval_advtrain:
        #r = attackvqa.evaluate(attackvqa.val_loader)
        # you can save result by calling:
        attackvqa.save_result_json(attackvqa.val_loader)
    if args.test_advtrain:
        # r = attackvqa.evaluate(attackvqa.val_loader, has_answers=False)
        attackvqa.save_result_json(attackvqa.val_loader, has_answers=False)


if __name__ == '__main__':
    main()