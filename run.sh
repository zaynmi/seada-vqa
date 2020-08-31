#python new_main.py --attack_only --attack_mode v --attack_al pgd --alpha 0.5 --iteration 6 --epsilon 5 --attacked_checkpoint logs/bs256.pth
#python main.py --advtrain --attack_mode vq --attack_al pgd,sea  --resume /home/tang/attack_on_VQA2.0-Recent-Approachs-2018/logs/baseline_10.pth
#python main.py --attack_only --attack_mode q --attack_al sea --attacked_checkpoint /home/tang/attack_on_VQA2.0-Recent-Approachs-2018/logs/bs256.pth --paraphrase_data train

python main.py --eval_advtrain --checkpoint logs/baseline_train_pgd,sea_vq_e0.3_it2_a0.5_w1_ad10_ld15_ade15_fr1.pth --attack_al ifgsm --attack_mode v --attacked_checkpoint /home/tang/attack_on_VQA2.0-Recent-Approachs-2018/logs/bs256.pth