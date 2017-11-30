import numpy as np
from PyNomaly import loop

prods = open('./prod.txt').readlines()
feats = open('./featprod_twitter.txt').readlines()

target_idx = 7
target = prods[target_idx].split('\x01')[5]
print(prods[target_idx].split('\x01')[6])

target_feats = list()
target_idcs = list()

for i, j in enumerate(prods):
    if j.split('\x01')[5] == target:
        target_feats.append([float(f) for f in feats[i].strip().split()])
        target_idcs.append(i)
target_feats = np.array(target_feats)
target_feats = target_feats[:1000]
scores = loop.LocalOutlierProbability(target_feats).fit()
sorted_idx = np.argsort(scores)
for i in range(10):
    print('Top %d(%f): ' % (i, scores[sorted_idx[i]]))
    print(prods[target_idcs[sorted_idx[i]]])
for i in range(scores.shape[0] - 20, scores.shape[0]):
    print('Low %d(%f): ' % (i, scores[sorted_idx[i]]))
    print(prods[target_idcs[sorted_idx[i]]])
