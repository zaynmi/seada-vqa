import json
ori_q = json.load(open('data/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
q_adv = json.load(open('data/v2_OpenEnded_mscoco_train2014_questions_adv07.json', 'r'))

ques_dict = {}
for q in q_adv['questions']:
    ques_dict[q['question_id']] = q
sorted_q = []
for q in ori_q['questions']:
    if q['question_id'] in ques_dict.keys():
        sorted_q.append(ques_dict[q['question_id']])
        print('#'*10)
        print('Original: %s' % q['question'])
        print('Adv:      %s' % ques_dict[q['question_id']]['question'])
    else:
        sorted_q.append(q)
sorted_q = {'questions': sorted_q}
with open('data/v2_OpenEnded_mscoco_train2014_questions_adv7.json', 'w') as f:
     json.dump(sorted_q, f)

print(len(sorted_q['questions']))
