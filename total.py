import numpy as np
from PyNomaly import loop
from subprocess import check_call

options = list()
for model_type in ('skipgram', 'cbow'):
    for pretrained in ('wiki', 'none'):
        # for dim in (100, 300, 500):
        for dim in (300, 500):
            if pretrained == 'wiki' and (dim == 100 or dim == 500): continue
            #for lr in (0.02, 0.05, 0.1):
            #for lr in (0.02, 0.05):
            for lr in (0.02,):
                for epoch in (5, 10):
                #for epoch in (5,):
                    # Model Generation
                    model_name = 'model_%s_%s_%d_%.2f_%d' % (
                            model_type,
                            pretrained,
                            dim,
                            lr,
                            epoch
                        )
                    gen_model_args = [
                        'fastText/fasttext',
                        model_type,
                        '-input', 'onlyprod_twitter_total.txt',
                        '-output', model_name,
                        '-dim', str(dim),
                        '-lr', str(lr),
                        '-epoch', str(epoch)
                    ]
                    if pretrained == 'wiki':
                        gen_model_args.append('-pretrainedVectors')
                        gen_model_args.append('wiki.ko.vec')
                    ret = check_call(gen_model_args)
                    if ret != 0:
                        print('Error when training model [%s]' % model_name)
                        continue

                    # Vector calculation
                    ret = check_call(['fastText/fasttext', 'print-sentence-vectors', model_name + '.bin'], stdin=open('onlyprod_twitter.txt'), stdout=open('featprod.txt', 'w'))
                    if ret != 0:
                        print('Error when generating vectors with model [%s]' % model_name)
                        continue

                    prods = open('./prod.txt').readlines()
                    feats = open('./featprod.txt').readlines()

                    target_idx = 1
                    target = prods[target_idx].split('\x01')[5]

                    target_feats = list()
                    target_idcs = list()
                    fake_idcs = list()

                    # NORMAL COUNT: Maximum 1000
                    for i, j in enumerate(prods):
                        if j.split('\x01')[5] == target:
                            target_feats.append([float(f) for f in feats[i].strip().split()])
                            target_idcs.append(i)
                            if len(target_idcs) >= 1000:
                                break

                    # FAKE COUNT: 50
                    fake_start = len(target_idcs)
                    for i in range(len(prods)):
                        if not i in target_idcs:
                            fake_idcs.append(i)
                            target_feats.append([float(f) for f in feats[i].strip().split()])
                            if len(fake_idcs) >= 50:
                                break

                    target_feats = np.array(target_feats)
                    scores = loop.LocalOutlierProbability(target_feats).fit()
                    sorted_idx = np.argsort(scores)

                    found_fakes_cnt = 0

                    for i in range(scores.shape[0] - 100, scores.shape[0]):
                        if sorted_idx[i] >= fake_start:
                            found_fakes_cnt += 1

                    print('Model [%20s]: Caught [%2d/50]' % (model_name, found_fakes_cnt))
                    check_call(['rm', model_name + '.bin'])
                    check_call(['rm', model_name + '.vec'])
