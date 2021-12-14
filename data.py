'''
prepare the triples for few-shot training
'''
import numpy as np
from collections import defaultdict
import json
import os
from tqdm import tqdm


def build_vocab(dataset, noise_rate):
    rels = set()
    ents = set()
    KG = []

    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]
            rels.add(rel)
            rels.add(rel + '_inv')
            ents.add(e1)
            ents.add(e2)

    KG = []

    with open(dataset + '_' + str(noise_rate) + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]
            KG.append([e1, rel, e2])

    # entid = json.load(open(dataset + '/ent2ids'))
    # relationid=json.load(open(dataset + '/relation2ids'))

    relationid = {}
    for idx, item in enumerate(list(rels)):
        relationid[item] = idx
    entid = {}
    for idx, item in enumerate(list(ents)):
        entid[item] = idx

    print('rel len', len(relationid))
    print('ent len', len(entid))

    json.dump(relationid, open(dataset + '_' + str(noise_rate) + '/relation2ids', 'w'))
    json.dump(entid, open(dataset + '_' + str(noise_rate) + '/ent2ids', 'w'))

    train2id = open(dataset + '_' + str(noise_rate) + '/train2id.txt', 'w')
    train2id.write(str(len(KG)) + '\n')
    for triple in KG:
        train2id.write('{} {} {}\n'.format(entid[triple[0]], entid[triple[2]], relationid[triple[1]]))

    entity2id = open(dataset + '_' + str(noise_rate) + '/entity2id.txt', 'w')
    entity2id.write(str(len(entid)) + '\n')
    for entity, id in entid.items():
        entity2id.write('{} {}\n'.format(entity, id))

    rel2id = open(dataset + '_' + str(noise_rate) + '/relation2id.txt', 'w')
    rel2id.write(str(len(relationid)) + '\n')
    for rel, id in relationid.items():
        rel2id.write('{} {}\n'.format(rel, id))


def freq_rel_triples(dataset):
    known_rels = defaultdict(list)
    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            e1, rel, e2 = line.split()
            known_rels[rel].append([e1, rel, e2])

    train_tasks = json.load(open(dataset + '/train_tasks.json'))

    for key, triples in train_tasks.items():
        known_rels[key] = triples

    json.dump(known_rels, open(dataset + '/known_rels.json', 'w'))


def for_filtering(dataset, save=False):
    e1rel_e2 = defaultdict(list)
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    test_tasks = json.load(open(dataset + '/test_tasks.json'))
    few_triples = []
    for _ in (list(train_tasks.values()) + list(dev_tasks.values()) + list(test_tasks.values())):
        few_triples += _
    for triple in few_triples:
        e1, rel, e2 = triple
        e1rel_e2[e1 + rel].append(e2)
    if save:
        json.dump(e1rel_e2, open(dataset + '/e1rel_e2.json', 'w'))


def transfer(dataset, noise_rate, train_keep_ratio=None):
    dev_tasks = json.load(open(dataset + '/dev_tasks.json'))
    test_tasks = json.load(open(dataset + '/test_tasks.json'))

    json.dump(dev_tasks, open(dataset + '_' + str(noise_rate) + '/dev_tasks.json', 'w'))

    # if train_keep_ratio==None and dataset!='Wiki':
    json.dump(test_tasks, open(dataset + '_' + str(noise_rate) + '/test_tasks.json', 'w'))

    try:
        rel2can = json.load(open(dataset + '/rel2candidates.json'))
        json.dump(rel2can, open(dataset + '_' + str(noise_rate) + '/rel2candidates.json', 'w'))
    except:
        pass


def build_all_train_tasks(dataset):
    rels = defaultdict(list)
    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]
            rels[rel].append((e1, rel, e2))
    print(len(rels))

    json.dump(rels, open(dataset + '/train_tasks_all.json', 'w'))


def build_noise_data(dataset, noise_rate):
    from numpy.random import RandomState

    total_pos = set()
    rel_e1_appear = defaultdict(list)  # key:rel value:e1 that appears in rel
    rel_e2_appear = defaultdict(list)

    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]

            total_pos.add(e1 + rel + e2)
            rel_e1_appear[rel].append(e1)
            rel_e2_appear[rel].append(e2)

    negative = []
    rd = RandomState(1)

    for key, value in rel_e1_appear.items():
        if len(value) < 2: print(key, value)

    for key, value in rel_e2_appear.items():
        if len(value) < 2: print(key, value)

    f_new = open(dataset + '_' + str(noise_rate) + '/path_graph', 'w')

    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        f_new.writelines(lines)
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]

            if rd.random() < noise_rate:

                if rd.random() > 0.5:
                    if len(rel_e1_appear[rel]) == 1: continue
                    count = 0
                    while True:
                        count += 1
                        new_e1 = rd.choice(rel_e1_appear[rel])
                        if new_e1 + rel + e2 not in total_pos or count == 20: break
                    if count == 20:
                        # print('skipped')
                        continue
                    negative.append(','.join([new_e1, rel, e2]))
                else:
                    if len(rel_e2_appear[rel]) == 1: continue
                    count = 0
                    while True:
                        count += 1
                        new_e2 = rd.choice(rel_e2_appear[rel])
                        if e1 + rel + new_e2 not in total_pos or count == 20: break
                    if count == 20:
                        # print('skipped')
                        continue
                    negative.append(','.join([e1, rel, new_e2]))

    print('total triple num', len(total_pos))
    print('negative triple num', len(negative))
    for nega in negative:
        f_new.write('\t'.join(nega.split(',')) + '\n')


def build_noise_train(dataset, noise_rate):
    from numpy.random import RandomState
    rd = RandomState(1)

    train_task = json.load(open(dataset + '/train_tasks.json'))

    noise_train_task = {}

    for rel, triples in train_task.items():
        pos_triples = [','.join(triple) for triple in triples]
        e1s = [triple[0] for triple in triples]
        e2s = [triple[2] for triple in triples]
        negs = []

        for triple in triples:
            if rd.random() < noise_rate:
                count = 0
                if rd.random() > 0.5:
                    while True:
                        count += 1
                        new_e1 = rd.choice(e1s)
                        if ','.join([new_e1, triple[1], triple[2]]) not in pos_triples or count == 100: break
                    if count != 100:
                        negs.append([new_e1, triple[1], triple[2]])
                else:
                    while True:
                        count += 1
                        new_e2 = rd.choice(e2s)
                        if ','.join([triple[0], triple[1], new_e2]) not in pos_triples or count == 100: break
                    if count != 100:
                        negs.append([triple[0], triple[1], new_e2])

        triples.extend(negs)
        noise_train_task[rel] = triples
    json.dump(noise_train_task, open(dataset + '_' + str(noise_rate) + '/train_tasks.json', 'w'))


def build_noise_new_setting(dataset, noise_rate, use_all_random=True, train_keep_ratio=None):
    from numpy.random import RandomState

    total_pos = set()
    rel_e1_appear = defaultdict(list)  # key:rel value:e1 that appears in rel
    rel_e2_appear = defaultdict(list)
    total_ent = set()

    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]

            if use_all_random:
                total_ent.add(e1)
                total_ent.add(e2)

            total_pos.add(e1 + rel + e2)
            rel_e1_appear[rel].append(e1)
            rel_e2_appear[rel].append(e2)
    total_ent = list(total_ent)

    original_total = len(total_pos)

    negative = []
    rd = RandomState(1)

    for key, value in rel_e1_appear.items():
        if len(value) < 2: print(key, value)

    for key, value in rel_e2_appear.items():
        if len(value) < 2: print(key, value)

    f_new = open(dataset + '_' + str(noise_rate) + '/path_graph', 'w')
    total_count = 0

    with open(dataset + '/path_graph') as f:
        # lines = f.readlines()
        # f_new.writelines(lines)
        for line in tqdm(lines):
            line = line.rstrip()
            rel = line.split('\t')[1]
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]

            count = 0
            a = rd.random()
            if a < noise_rate:

                if rd.random() > 0.5:
                    if len(rel_e1_appear[rel]) == 1: continue
                    while True:
                        count += 1
                        if dataset != 'Wiki':
                            if use_all_random:
                                new_e1 = rd.choice(total_ent)
                            else:
                                new_e1 = rd.choice(rel_e1_appear[rel])
                        elif dataset == 'Wiki':
                            if use_all_random:
                                # new_e1=rd.choice(total_ent)
                                new_e1 = total_ent[min(round(rd.random() * len(total_ent)), len(total_ent) - 1)]
                            else:
                                new_e1 = rel_e1_appear[rel][round(a / noise_rate * len(rel_e1_appear[rel])) - 1]
                        if new_e1 + rel + e2 not in total_pos or count == 20: break
                    if count == 20:
                        # print('skipped')
                        f_new.write('\t'.join([e1, rel, e2]) + '\n')
                        total_count += 1
                        continue
                    total_pos.add(new_e1 + rel + e2)
                    negative.append(','.join([new_e1, rel, e2]))
                else:
                    if len(rel_e2_appear[rel]) == 1: continue
                    while True:
                        count += 1
                        if dataset != 'Wiki':
                            if use_all_random:
                                new_e2 = rd.choice(total_ent)
                            else:
                                new_e2 = rd.choice(rel_e2_appear[rel])
                        elif dataset == 'Wiki':
                            if use_all_random:
                                # new_e2=rd.choice(total_ent)
                                new_e2 = total_ent[min(round(rd.random() * len(total_ent)), len(total_ent) - 1)]
                            else:
                                new_e2 = rel_e2_appear[rel][round(a / noise_rate * len(rel_e2_appear[rel])) - 1]
                        if e1 + rel + new_e2 not in total_pos or count == 20: break
                    if count == 20:
                        # print('skipped')
                        f_new.write('\t'.join([e1, rel, e2]) + '\n')
                        total_count += 1
                        continue
                    total_pos.add(e1 + rel + new_e2)
                    negative.append(','.join([e1, rel, new_e2]))

            else:
                f_new.write('\t'.join([e1, rel, e2]) + '\n')
                total_count += 1

    print('total pos triple num', original_total)
    print('negative triple num', len(negative))
    for nega in negative:
        f_new.write('\t'.join(nega.split(',')) + '\n')
        total_count += 1
    print('total final triple num', total_count)
    print('final neg ratio', len(negative) / total_count)

    # ----------------------------------------

    train_task = json.load(open(dataset + '/train_tasks.json'))
    test_task = json.load(open(dataset + '/test_tasks.json'))

    # if dataset=='Wiki':
    #    new_test_task={}
    #    for key in test_task:
    #        if len(new_test_task)<22:
    #            new_test_task[key]=test_task[key]
    #        else:
    #            train_task[key]=test_task[key]
    #    test_task=new_test_task

    if train_keep_ratio != None:
        train_keep_num = train_keep_ratio
    else:
        train_keep_num = None

    noise_train_task = {}

    for rel, triples in train_task.items():
        if train_keep_ratio != None and len(noise_train_task) == train_keep_num:
            test_task[rel] = train_task[rel]
            continue

        pos_triples = [','.join(triple) for triple in triples]

        if use_all_random:
            e1s = total_ent
            e2s = total_ent
        else:
            e1s = [triple[0] for triple in triples]
            e2s = [triple[2] for triple in triples]
        negs = []

        for triple in triples:
            count = 0
            if rd.random() < noise_rate:
                if rd.random() > 0.5:
                    while True:
                        count += 1
                        if dataset != 'Wiki':
                            new_e1 = rd.choice(e1s)
                        else:
                            new_e1 = total_ent[min(round(rd.random() * len(total_ent)), len(total_ent) - 1)]
                        if ','.join([new_e1, triple[1], triple[2]]) not in pos_triples or count == 100: break
                    if count != 100:
                        negs.append([new_e1, triple[1], triple[2]])
                else:
                    while True:
                        count += 1
                        if dataset != 'Wiki':
                            new_e2 = rd.choice(e2s)
                        else:
                            new_e2 = total_ent[min(round(rd.random() * len(total_ent)), len(total_ent) - 1)]
                        if ','.join([triple[0], triple[1], new_e2]) not in pos_triples or count == 100: break
                    if count != 100:
                        negs.append([triple[0], triple[1], new_e2])
            if count == 0 or count == 100:
                negs.append(triple)

        noise_train_task[rel] = negs

    json.dump(noise_train_task, open(dataset + '_' + str(noise_rate) + '/train_tasks.json', 'w'))

    # if train_keep_ratio!=None or dataset!='NELL':
    # json.dump(test_task,open(dataset+'_'+str(noise_rate) + '/test_tasks.json','w'))


def create_noise_data(dataset, noise_rate, use_all_random=True, train_keep_ratio=None):
    import os
    os.mkdir(dataset + '_' + str(noise_rate))
    # build_noise_data(dataset, noise_rate)
    # build_noise_train(dataset, noise_rate)
    build_noise_new_setting(dataset, noise_rate, use_all_random=use_all_random, train_keep_ratio=train_keep_ratio)
    transfer(dataset, noise_rate, train_keep_ratio=train_keep_ratio)
    # use original dataset to build vocab in case that due to noise some entities disappear
    build_vocab(dataset, noise_rate)
    for_filtering(dataset + '_' + str(noise_rate), save=True)


def create_noisy_test_support(dataset, noise_rate, support_num, mode='test'):
    from numpy.random import RandomState
    rd = RandomState(1)
    test_tasks = json.load(open(dataset + '/{}_tasks.json'.format(mode)))

    total_ent = set()

    with open(dataset + '/path_graph') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            e1 = line.split('\t')[0]
            e2 = line.split('\t')[2]

            total_ent.add(e1)
            total_ent.add(e2)

    total_ent = list(total_ent)

    e1s = total_ent
    e2s = total_ent

    noise_test_task = {}

    for rel, triples in test_tasks.items():
        pos_triples = [','.join(triple) for triple in triples]

        negs = []

        for triple in triples[:support_num]:
            count = 0
            if rd.random() < noise_rate:
                if rd.random() > 0.5:
                    while True:
                        count += 1
                        if dataset != 'Wiki':
                            new_e1 = rd.choice(e1s)
                        else:
                            new_e1 = total_ent[min(round(rd.random() * len(total_ent)), len(total_ent) - 1)]

                        if ','.join([new_e1, triple[1], triple[2]]) not in pos_triples or count == 100: break
                    if count != 100:
                        negs.append([new_e1, triple[1], triple[2]])
                else:
                    while True:
                        count += 1
                        if dataset != 'Wiki':
                            new_e2 = rd.choice(e2s)
                        else:
                            new_e2 = total_ent[min(round(rd.random() * len(total_ent)), len(total_ent) - 1)]

                        if ','.join([triple[0], triple[1], new_e2]) not in pos_triples or count == 100: break
                    if count != 100:
                        negs.append([triple[0], triple[1], new_e2])
            if count == 0 or count == 100:
                negs.append(triple)

        noise_test_task[rel] = negs + triples[support_num:]

        json.dump(noise_test_task, open(dataset + '/{}_tasks_noisy.json'.format(mode), 'w'))


if __name__ == '__main__':

    # build_all_train_tasks('NELL')
    # build_noise_data('NELL',0.4)
    # build_vocab('NELL_0.4')
    # build_noise_train('NELL_0.4',0.4)
    # for_filtering('NELL',True)
    # create_noise_data('NELL', -1, train_keep_ratio=40)
    # create_noise_data('NELL',0.1,train_keep_ratio=40)
    # create_noise_data('NELL',0.2,train_keep_ratio=40)
    # create_noise_data('NELL',0.3,train_keep_ratio=40)
    # create_noise_data('Wiki',-1)
    # create_noise_data('Wiki',0.1)
    # create_noise_data('FB15K',0.05)
    # create_noise_data('FB15K',0.1)
    # create_noise_data('FB15K',0.2)
    # create_noise_data('FB15K',0.3)

    # build_vocab('NELL')
    # build_noise_new_setting('NELL',0.7)

    dataset='Wiki'
    for ratio in [0.05,0.1,0.15,0.2]:
        create_noise_data(dataset,ratio)

    for dataset, noise_rate in zip(
            ['Wiki', 'Wiki_0.05', 'Wiki_0.1', 'Wiki_0.15', 'Wiki_0.2', 'FB15K_0.0_t33', 'FB15K_0.05_t33',
             'FB15K_0.1_t33', 'FB15K_0.15_t33', 'FB15K_0.2_t33', 'NELL', 'NELL_0.05', 'NELL_0.1', 'NELL_0.15',
             'NELL_0.2'], [-1, 0.05, 0.1, 0.15, 0.2, -1, 0.05, 0.1, 0.15, 0.2, -1, 0.05, 0.1, 0.15, 0.2]):
        create_noisy_test_support(dataset, noise_rate, support_num=5,mode='dev')
