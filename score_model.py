import numpy as np
from PyNomaly import loop
from subprocess import check_call

model_name = 'model_twitter_250000.bin'
check_call(['fastText/fasttext', 'print-sentence-vectors', model_name], stdin=open('onlyprod_twitter.txt'), stdout=open('featprod_twitter.txt', 'w'))

prods = open('./prod.txt').readlines()
feats = open('./featprod_twitter.txt').readlines()

target_idx = 0
target = prods[target_idx].split('\x01')[5]
print(prods[target_idx].split('\x01')[6])

target_feats = list()
target_idcs = list()
fake_idcs = list()

# NORMAL COUNT: Maximum 1000
for i, j in enumerate(prods):
    if j.split('\x01')[5] == target:
        target_feats.append([float(f) for f in feats[i].strip().split()])
        target_idcs.append(i)
        if len(target_idcs) > 1000:
            break

# FAKE COUNT: 25
fake_start = len(target_idcs)
for i in range(len(prods)):
    if not i in target_idcs:
        fake_idcs.append(i)
        target_feats.append([float(f) for f in feats[i].strip().split()])
        if len(fake_idcs) > 25:
            break

target_feats = np.array(target_feats)
scores = loop.LocalOutlierProbability(target_feats).fit()
sorted_idx = np.argsort(scores)

found_fakes_cnt = 0

for i in range(scores.shape[0] - 50, scores.shape[0]):
    if sorted_idx[i] >= fake_start:
        found_fakes_cnt += 1

print('Model [%20s]: Caught [%2d/25]' % (model_name, found_fakes_cnt))
