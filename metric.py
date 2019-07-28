import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk import ngrams
import os

f = open('E:/nlp/doctor/samples.json', encoding='utf-8')
line = f.readline()
avglen_ques = 0
avglen_ans = 0
pair_count = 0
avg_ans = 0
ans_len = 0
all_ans_count = 0
queslen_dic = [0,0,0,0,0,0]
anslen_dic = [0,0,0,0,0,0]
f_ques = open('E:/nlp/doctor/ques', 'w', encoding='utf-8')
f_ans = open('E:/nlp/doctor/ans', 'w', encoding='utf-8')
f_rans = open('E:/nlp/doctor/rans', 'w', encoding='utf-8')
f_all_rans = open('E:/nlp/doctor/all_rans', 'w', encoding='utf-8')
f_all_rques = open('E:/nlp/doctor/all_rques', 'w', encoding='utf-8')
ans_words = []
ques_words = []
related_words = []
while line:
    file = json.loads(line, encoding='utf-8')
    ques_title = file['ques_title'].split(' ')
    ques_content = file['ques_content'].split(' ')
    ques = ques_title + ques_content

    for w in ques:
        ques_words.append(w)
        f_ques.write(w + ' ')
    f_ques.write('\n')

    ans_contents = file['ans_contents']
    ans_count = 0
    f_ans.write(ans_contents[0])
    for ans in ans_contents:
        ans = ans.split(' ')
        for w in ans:
            ans_words.append(w)
        ans_count += 1
        ans_len += len(ans)

        if len(ans) <= 40:
            anslen_dic[0] += 1
        elif len(ans) <= 50:
            anslen_dic[1] += 1
        elif len(ans) <= 60:
            anslen_dic[2] += 1
        elif len(ans) <= 70:
            anslen_dic[3] += 1
        elif len(ans) <= 80:
            anslen_dic[4] += 1
        else:
            anslen_dic[5] += 1


    f_ans.write('\n')
    all_ans_count += ans_count
    related = file['related']
    relevant_ans = related[0]['ans_contents']
    f_rans.write(relevant_ans[0])
    for i in range(0,5):
        ques = related[i]['ques_title'].split(' ') + related[i]['ques_content'].split(' ')
        for w in ques:
            f_all_rques.write(w + ' ')
            related_words.append(w)
        ans = related[i]['ans_contents']
        for a in ans:
            f_all_rans.write(a + ' ')
            for w in a:
                related_words.append(w)

    f_all_rques.write('\n')
    f_rans.write('\n')
    f_all_rans.write('\n')


    ques_tmp = len(ques_content) + len(ques_title)
    avglen_ques += ques_tmp

    if ques_tmp <= 20:
        queslen_dic[0] += 1
    elif ques_tmp <= 30:
        queslen_dic[1] += 1
    elif ques_tmp <= 40:
        queslen_dic[2] += 1
    elif ques_tmp <= 50:
        queslen_dic[3] += 1
    elif ques_tmp <= 60:
        queslen_dic[4] += 1
    else:
        queslen_dic[5] += 1

    pair_count += 1
    line = f.readline()

ques_words = set(ques_words)
ans_words = set(ans_words)
related_words = set(related_words)

f.close()
f_ans.close()
f_rans.close()
f_ques.close()
f_all_rans.close()
f_all_rques.close()

avg_ans = all_ans_count / pair_count
avglen_ques = avglen_ques / pair_count
avglen_ans = ans_len / all_ans_count

print("平均Q句长：" + str(avglen_ques))
print("平均A句长：" + str(avglen_ans))
print("平均回答数：" + str(avg_ans))


fig, axes = plt.subplots(2, 1)
plt.subplots_adjust(hspace=0.5)

data = pd.Series(queslen_dic, index=list(['<20', '20-30', '30-40', '40-50', '50-60', '>60']))
data.plot(title='queslen', kind='barh', ax=axes[0], color='k', alpha=0.7)
data = pd.Series(anslen_dic, index=list(['<40', '40-50', '50-60', '60-70', '70-80', '>80']))
data.plot(title='anslen', kind='barh', ax=axes[1], color='k', alpha=0.7)

fig.savefig('p1.png')

n_grams_range = [1, 2, 3, 4]
all_summary_novel_ngram_rate_list_1 = [[] for _ in n_grams_range]
all_summary_novel_ngram_rate_list_2 = [[] for _ in n_grams_range]
for n in n_grams_range:
    content_ng = list(ngrams(ques_words, n))
    summary_ng = list(ngrams(ans_words, n))
    if len(content_ng) == 0 or len(summary_ng) == 0:
        continue
    novel_counter = 0
    for w in summary_ng:

        if w not in content_ng:
            novel_counter += 1
    all_summary_novel_ngram_rate_list_1[n - 1].append(novel_counter / len(summary_ng))

    content_ng = list(ngrams(related_words, n))
    summary_ng = list(ngrams(ans_words, n))
    if len(content_ng) == 0 or len(summary_ng) == 0:
        continue
    novel_counter = 0
    for w in summary_ng:

        if w not in content_ng:
            novel_counter += 1
    all_summary_novel_ngram_rate_list_2[n - 1].append(novel_counter / len(summary_ng))

print("A-Q:")
for n in n_grams_range:
    print('%d novel grams %f' % (n, np.mean(all_summary_novel_ngram_rate_list_1[n - 1])))
print("A-Q'-A':")
for n in n_grams_range:
    print('%d novel grams %f' % (n, np.mean(all_summary_novel_ngram_rate_list_2[n - 1])))


import subprocess
bleu1 = subprocess.getoutput("perl E:/nlp/doctor/multi-bleu-yiping.perl E:/nlp/doctor/ans<E:/nlp/doctor/rans")
bleu2 = subprocess.getoutput("perl E:/nlp/doctor/multi-bleu-yiping.perl E:/nlp/doctor/ans<E:/nlp/doctor/all_rans")
bleu3 = subprocess.getoutput("perl E:/nlp/doctor/multi-bleu-yiping.perl E:/nlp/doctor/ques<E:/nlp/doctor/all_rques")
print("A1:\n" + bleu1)
print("all related ans:\n" + bleu2)
print("all related ques:\n" + bleu3)

os.remove("E:/nlp/doctor/ques")
os.remove("E:/nlp/doctor/ans")
os.remove("E:/nlp/doctor/rans")
os.remove("E:/nlp/doctor/all_rans")
os.remove("E:/nlp/doctor/all_rques")