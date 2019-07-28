p1 = '/home1/lmz/pointer-generator-master/directory_d/myexperiment/decode_test_400maxenc_1beam_35mindec_100maxdec_ckpt-470296/decoded/'
p2 = '/home1/lmz/pointer-generator-master/directory_d/myexperiment/decode_test_400maxenc_1beam_35mindec_100maxdec_ckpt-470296/reference/'
p3 = '/home1/lmz/pointer-generator-master/directory_d/decoded.txt'
p4 = '/home1/lmz/pointer-generator-master/directory_d/reference.txt'

with open(p3, 'w',encoding='utf-8') as fw:
    for i in range(0, 5000):
        fname = p1 + ("%06d" % i) + '_decoded.txt'
        f1 = open(fname, 'r', encoding='utf-8')
        str = f1.readline()
        if '\n' in str:
            str = str.strip('\n')
        fw.write(str + '\n')

with open(p4, 'w',encoding='utf-8') as fw:
    for i in range(0, 5000):
        fname = p2 + ("%06d" % i) + '_reference.txt'
        f2 = open(fname, 'r', encoding='utf-8').readlines()
        for w in f2:
            w = w.strip()
            fw.write(w + ' ')
        fw.write('\n')