import json

qtrainadv = json.load(open('data/v2_OpenEnded_mscoco_train2014_questions_adv.json', 'r'))
qvaladv = json.load(open('data/v2_OpenEnded_mscoco_val2014_questions_adv.json', 'r'))
qtrainval = json.load(open('data/v2_OpenEnded_mscoco_trainval2014_questions.json', 'r'))

qmerge = {'questions': qtrainadv['questions'] + qvaladv['questions']}

print(len(qtrainval['questions']))
print(len(qmerge['questions']))

with open('data/v2_OpenEnded_mscoco_trainval2014_questions_adv.json', 'w') as f:
    json.dump(qmerge, f)

