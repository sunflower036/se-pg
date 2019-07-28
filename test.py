import subprocess
line = subprocess.getoutput("perl E:/nlp/doctor/multi-bleu-yiping.perl E:/nlp/doctor/ques<E:/nlp/doctor/all_rques")
print(line)

# BLEU = 4.1779, 27.2862, 6.1122, 1.8331, 0.9966 (BP=1.000, ratio=1.018, hyp_len=11438, ref_len=11232)