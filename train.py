import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from numpy.random import RandomState
import pickle as pkl
from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import os

if_cuda = torch.cuda.is_available()
print('CUDA', if_cuda)


class one_task(object):
    def __init__(self, support_set, query_set, query_label):
        self.support_set = support_set  # [C-way, K-shot, 2]
        self.query_set = query_set  # [C-way * query_size, 2]
        self.query_label = query_label.cuda() if if_cuda else query_label  # [C_way * query_size] range [0, C_way)
        self.C_way = support_set.size()[0]
        self.K_shot = support_set.size()[1]

    def get_left_and_right(self, set):
        return set[:, :, 0], set[:, :, 1]


class train_dataset(object):
    def __init__(self, dataset, C_way, K_shot, symbol2id, query_size=10, iter_num=100, train_tasks_all_min_num=50):
        self.train_tasks = json.load(open(dataset + '/train_tasks.json'))
        if K_shot != 5:
            self.test_tasks = json.load(open(dataset + '/test_tasks_noisy_{}.json'.format(K_shot)))
            self.dev_tasks = json.load(open(dataset + '/dev_tasks_noisy_{}.json'.format(K_shot)))
        else:
            self.test_tasks = json.load(open(dataset + '/test_tasks_noisy.json'))
            self.dev_tasks = json.load(open(dataset + '/dev_tasks_noisy.json'))

        self.relations = []
        for key, value in self.train_tasks.items():
            if len(value) > train_tasks_all_min_num:
                self.relations.append(key)
        self.C_way = C_way
        self.K_shot = K_shot
        self.symbol2id = symbol2id
        self.query_size = query_size
        self.iter_num = iter_num

    def __len__(self):
        return self.iter_num

    def get_test_task(self, mode='test'):

        if mode == 'test':
            task = self.test_tasks
        else:
            task = self.dev_tasks

        support_set = []
        query_set = []
        query_label = []
        for i, key in enumerate(list(task.keys())):
            train_triples = task[key]
            support_triples = train_triples[:self.K_shot]
            support_set.append([[self.symbol2id[triple[0]], self.symbol2id[triple[2]]] for triple in
                                support_triples])  # [C_way, K_shot, 2]

            query_triples = train_triples[self.K_shot:]

            query_set.extend([[self.symbol2id[triple[0]], self.symbol2id[triple[2]]] for triple in query_triples])
            query_label.extend([i] * len(query_triples))

        return one_task(torch.tensor(support_set).cuda() if if_cuda else torch.tensor(support_set),
                        torch.tensor(query_set).cuda() if if_cuda else torch.tensor(query_set),
                        torch.tensor(query_label))

    def getitem(self, idx):
        #get a task for tranining
        if idx == 0: np.random.shuffle(self.relations)

        # rd = RandomState(idx)
        rd = np.random

        relations = rd.choice(self.relations, size=self.C_way,replace=False)
        support_set = []
        query_set = []
        query_label = []
        for i, key in enumerate(relations):
            train_triples = self.train_tasks[key]
            rd.shuffle(train_triples)

            support_triples = train_triples[:self.K_shot]
            support_set.append([[self.symbol2id[triple[0]], self.symbol2id[triple[2]]] for triple in
                                support_triples])  # [C_way, K_shot, 2]
            query_triples = train_triples[self.K_shot:]
            if len(query_triples) < self.query_size:
                if len(query_triples) == 0: raise ValueError('relation samples less than K-shot')
                query_triples = [one.split(' ') for one in
                                 rd.choice([' '.join(one) for one in query_triples], size=self.query_size)]

            else:
                query_triples = query_triples[:self.query_size]
            query_set.extend([[self.symbol2id[triple[0]], self.symbol2id[triple[2]]] for triple in query_triples])
            query_label.extend([i] * self.query_size)

        return one_task(torch.tensor(support_set).cuda() if if_cuda else torch.tensor(support_set),
                        torch.tensor(query_set).cuda() if if_cuda else torch.tensor(query_set),
                        torch.tensor(query_label))


class Trainer(object):

    def __init__(self, dataset, embed_model, C_way, K_shot, save_path, max_neighbor=50, test=False, weight_decay=0.0,
                 lr=0.001, random_embed=False, query_size=10, batch_num_max=500000, save_rel=False,
                 baseline_proto=False):
        self.dataset = dataset
        self.embed_model = embed_model
        self.C_way = C_way
        self.K_shot = K_shot
        self.save_path = save_path
        self.max_neighbor = max_neighbor
        self.if_test = test
        self.weight_decay = weight_decay
        self.lr = lr
        self.random_embed = random_embed

        self.save_rel = save_rel

        if self.if_test or self.random_embed:
            self.load_symbol2id()
            self.use_pretrain = False
        else:
            # load pretrained embedding
            self.load_embed()
            self.use_pretrain = True

        self.train_dataset = train_dataset(self.dataset, self.C_way, self.K_shot, self.symbol2id, query_size=query_size)

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        self.pad_id = self.num_symbols
        self.matcher = Matcher(self.symbol2vec, self.num_symbols, self.max_neighbor, self.C_way, self.K_shot,
                               fine_tune=False, baseline_proto=baseline_proto)
        if if_cuda: self.matcher.cuda()

        self.batch_num_max = batch_num_max

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200000], gamma=0.5)

        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.num_rels = len(self.rel2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')


        logging.info('LOADING CANDIDATES ENTITIES')


    def load_embed(self):
        #load the pre-trained embeddings for relations and entities
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')


        ent_embed = np.array(json.load(open('{}/transe.json'.format(self.dataset)))['ent_embeddings.weight'])
        rel_embed = np.array(json.load(open('{}/transe.json'.format(self.dataset)))['rel_embeddings.weight'])


        print(ent_embed.shape)
        print(len(ent2id.keys()))

        assert ent_embed.shape[0] == len(ent2id.keys())
        assert rel_embed.shape[0] == len(rel2id.keys())
        i = 0
        embeddings = []
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(rel_embed[rel2id[key], :]))
        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(list(ent_embed[ent2id[key], :]))
        symbol_id['PAD'] = i
        embeddings.append(list(np.zeros((rel_embed.shape[1],))))
        embeddings = np.array(embeddings)

        print(embeddings.shape[0], len(symbol_id.keys()))

        assert embeddings.shape[0] == len(symbol_id.keys())

        self.symbol2id = symbol_id
        self.symbol2vec = embeddings

    def build_connection(self, max_=100):  # 100
        #build connections for neighbor information

        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors))
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        # json.dump(degrees, open(self.dataset + '/degrees', 'w'))

        return degrees

    def get_meta(self, ent_symbol_id):
        # for a single relation or a query set
        # only in self.connections the ent_id is different from its symbol_id
        ent_id = ent_symbol_id - self.num_rels

        connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in ent_id], axis=0)))
        degrees = Variable(torch.FloatTensor([self.e1_degrees[_.item()] for _ in ent_id]))

        if if_cuda:
            return connections.cuda(), degrees.cuda()

        else:
            return connections, degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path + '/{}.model'.format(self.dataset.strip('./')))

    def load(self):
        self.matcher.load_state_dict(torch.load(self.save_path))

    def train(self):
        #training process
        self.best_test_score = 0
        best_dev_score = 0
        losses = []
        accs = []
        test_losses = []
        test_accs = []
        self.matcher.train()
        # for task in Data.DataLoader(dataset=self.train_dataset, batch_size=1, shuffle=True, num_workers=1):
        for j in range(self.batch_num_max):
            task = self.train_dataset.getitem(j)

            meta_support_left = [self.get_meta(task.support_set[i, :, 0]) for i in
                                 range(
                                     task.C_way)]  # [2(connections & degrees), C_way, K_shot, left_neighbor_num, 2(rel & ent)], symbol_id
            meta_support_right = [self.get_meta(task.support_set[i, :, 1]) for i in
                                  range(task.C_way)]
            meta_query_left = self.get_meta(task.query_set[:, 0])  # [2, C_way * query_size, left_neighbor_num, 2]
            meta_query_right = self.get_meta(task.query_set[:, 1])

            query_scores, loss = self.matcher(task, meta_support_left, meta_support_right, meta_query_left,
                                              meta_query_right)


            losses.append(loss.item())
            acc = torch.eq(torch.argmax(query_scores, dim=-1), task.query_label).float().mean()
            accs.append(acc.item())

            logging.info(loss.item(), acc.item())

            if j % 50 == 0:
                print('Epoch {} loss: {:.4f} acc: {:.4f}'.format(j, loss.item(), acc.item()))

            self.optim.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(self.parameters, 5)

            self.optim.step()

            test_step = 50 if 'Wiki' not in self.dataset else 500

            if j % test_step == 0:
                _, dev_score = self.test('dev', require_rel_result=self.save_rel)
                if dev_score > best_dev_score:
                    best_dev_score = dev_score
                    self.rel_dev_best_acc = list(self.rel_acc)
                    test_loss, test_score = self.test(mode='test', require_rel_result=self.save_rel)
                    self.rel_test_best_acc = list(self.rel_acc)
                    test_losses.append(test_loss)
                    test_accs.append(test_score)
                    if test_score > self.best_test_score:
                        self.best_test_score = test_score
                        if 'Wiki' not in self.dataset:
                            # self.save()
                            pass
                        json.dump(self.best_test_score,
                                  open(self.save_path + '/{}_best_acc.json'.format(self.dataset.strip('./')),
                                       'w'))


        # return self.best_test_score
        return test_accs[-1]

    def test(self, mode='test', require_rel_result=False, require_v_values=False):

        with torch.no_grad():
            self.matcher.eval()
            task = self.train_dataset.get_test_task(mode=mode)

            meta_support_left = [self.get_meta(task.support_set[i, :, 0]) for i in
                                 range(
                                     task.C_way)]  # [2(connections & degrees), C_way, K_shot, left_neighbor_num, 2(rel & ent)], symbol_id
            meta_support_right = [self.get_meta(task.support_set[i, :, 1]) for i in
                                  range(task.C_way)]

            meta_query_left = self.get_meta(task.query_set[:, 0])  # [2, test_query_size , left_neighbor_num, 2]
            meta_query_right = self.get_meta(task.query_set[:, 1])

            query_scores, loss = self.matcher(task, meta_support_left, meta_support_right, meta_query_left,
                                              meta_query_right)



            acc = torch.eq(torch.argmax(query_scores, dim=-1), task.query_label).float().mean()

            logging.info(loss.item(), acc.item())

            print('{}: loss: {:.4f} acc: {:.4f}'.format(mode, loss.item(), acc.item()))

            if require_rel_result:

                predict = torch.argmax(query_scores, dim=-1)
                label = task.query_label
                acc_rel = [0 for i in range(task.C_way)]
                count = [0 for i in range(task.C_way)]
                # print(len(predict))
                for i in range(len(predict)):
                    if predict[i] == label[i]:
                        acc_rel[label[i]] += 1
                    count[label[i]] += 1

                self.rel_acc = [acc_rel[i] / count[i] for i in range(task.C_way)]



            self.matcher.train()

            return loss.item(), acc.item()


class Support_Encoder(nn.Module):

    def __init__(self):
        super(Support_Encoder, self).__init__()

    def forward(self):
        return


class Query_Encoder(nn.Module):

    def __init__(self):
        super(Query_Encoder, self).__init__()

    def forward(self):
        return


class Matcher(nn.Module):

    def __init__(self, emb, num_symbols, max_neighbor, C_way, K_shot, use_pretrain=True, drop_out_rate=0.2,
                 fine_tune=True, baseline_proto=False):
        super(Matcher, self).__init__()

        self.embed_dim = emb.shape[-1]
        self.symbol_emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx=num_symbols)
        self.max_neighbor = max_neighbor
        self.baseline_proto = baseline_proto
        self.proto_FC = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.relation_FC1 = nn.Linear(4 * self.embed_dim, 8)
        self.relation_FC2 = nn.Linear(8, 1)

        self.C_way = C_way
        self.K_shot = K_shot

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.att_W_key = nn.Linear(self.embed_dim, self.embed_dim)  # Project embedding of one query node
        self.att_W_query = nn.Linear(2 * self.embed_dim,
                                     self.embed_dim)  # project embedding of one relation and one node of neighbors

        self.sqrt_d = torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float))

        self.support_confidence_1 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.support_confidence_2 = nn.Linear(self.embed_dim, 1)

        self.W_energy = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)
        self.gamma = 0.2
        self.conf_att = nn.Linear(2 * self.embed_dim, 2 * self.embed_dim)

        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim))
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)

        num_head = 4
        num_layer = 3

        self.transformer_instance = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(2 * self.embed_dim, num_head, dropout=drop_out_rate), num_layer)

        self.transformer_relations = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(2 * self.embed_dim, num_head, dropout=drop_out_rate), num_layer)


        self.L = 20
        self.phi_1 = nn.Linear(self.L, self.L)
        self.phi_2 = nn.Linear(self.L, self.L)
        self.W_v = nn.Linear(self.L, self.L)
        self.W_y = nn.Linear(self.L, self.L)
        self.W_u = nn.Linear(self.L, 1)

        self.G_w = nn.Linear(8 * self.embed_dim, self.embed_dim)
        self.G_y = nn.Linear(self.embed_dim, self.embed_dim)
        self.G_u = nn.Linear(self.embed_dim, 1)
        self.p_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_abs = nn.Linear(self.embed_dim, 1)

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            # emb_np = np.loadtxt(embed_path)
            self.symbol_emb.weight.data.copy_(torch.from_numpy(emb))
            if not fine_tune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.dropout = nn.Dropout(drop_out_rate)

        self.query_encoder = Query_Encoder()
        self.support_encoder = Support_Encoder()

    def length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) >= length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)

        return mask.cuda() if if_cuda else mask

    def neighbor_encoder(self, connections, num_neighbors, entities_source, att=True):

        # connections: [K_shot, neighbor_num, 2(rel & ent)]
        # num_beighbors: [K_shot]
        # two_entities: [K_shot]

        use_neighbor_encoder = True

        embeddings = self.dropout(self.symbol_emb(entities_source))  # [K_shot,embed_dim]

        # baseline_proto = False
        if self.baseline_proto:
            return embeddings, 0

        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)

        rel_embeds = self.dropout(self.symbol_emb(relations))  # (K_shot, neighbor_num, embed_dim)
        ent_embeds = self.dropout(self.symbol_emb(entities))  # (K_shot, neighbor_num, embed_dim)


        average_confi = 0

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)

        if_confidence = False
        if if_confidence:
            ##produce TransE energy scores
            repeated_emb = embeddings.unsqueeze(1).repeat(
                [1, rel_embeds.shape[1], 1])  # (K_shot, neighbor_num, embed_dim)
            energy = torch.norm(ent_embeds + rel_embeds - repeated_emb, dim=-1, p=2)
            confidence = torch.exp(-energy)  # (K_shot, neighbor_num)

        if att:

            key_embeddings = self.att_W_key(embeddings).tanh()  # [K_shot, embed_dim]
            query_embeddings = self.att_W_query(concat_embeds).tanh()  # [K_shot, neighbor_num, embed_dim]

            att_values = torch.matmul(query_embeddings,
                                      key_embeddings.unsqueeze(-1)).squeeze() / self.sqrt_d  # [K_shot, neighbor_num]
            if if_confidence:
                att_values = att_values * confidence

            normalized_att = att_values.masked_fill(
                self.length_to_mask(num_neighbors.long(), max_len=self.max_neighbor),
                value=-1e9).softmax(-1)  # [K_shot, neighbor_num]

            # residue
            out = torch.matmul(normalized_att.unsqueeze(1), self.gcn_w(concat_embeds)).squeeze() + embeddings

        else:
            out = self.gcn_w(concat_embeds)
            out = torch.sum(out, dim=1)  # (K_shot, embed_dim)
            out = out / num_neighbors.unsqueeze(-1)

        if use_neighbor_encoder:
            return out.tanh(), average_confi
        else:
            return embeddings, average_confi

    def meta_support_encode(self, meta_support_left, meta_support_right, task):
        #aggregate information from neighbors
        support_left_degrees = torch.stack([one[1] for one in meta_support_left], dim=0)  # [C_way, K_shot]
        support_left_connetions = torch.stack([one[0] for one in meta_support_left],
                                              dim=0)  # [C_way, K_shot, num_neighbor, 2]
        support_right_degrees = torch.stack([one[1] for one in meta_support_right], dim=0)
        support_right_connetions = torch.stack([one[0] for one in meta_support_right], dim=0)

        results = [self.neighbor_encoder(support_left_connetions[i], support_left_degrees[i], task.support_set[i, :, 0])
                   for i in range(task.C_way)]
        support_left = torch.stack(
            [result[0] for result in results], dim=0)  # (C_way, K_shot, emb_dim)


        results = [
            self.neighbor_encoder(support_right_connetions[i], support_right_degrees[i], task.support_set[i, :, 1])
            for i in range(task.C_way)]
        support_right = torch.stack(
            [result[0] for result in results], dim=0)  # (C_way, K_shot, emb_dim)

        return support_left, support_right  # , confidence_left, confidence_right

    def meta_query_encode(self, meta_query_left, meta_support_right):
        pass

    def energy_func(self, ent_emb, rel_emb, if_query):
        # ent_emb: [N_1,N_2,```, 2*emb_size] rel_emb: [N_1,N_2,```,2*emb_size]

        if not if_query:
            return torch.matmul(self.W_energy(ent_emb).tanh().unsqueeze(-2),
                                rel_emb.unsqueeze(-1).tanh()).squeeze()  # [N]
        else:
            return torch.matmul(self.W_energy(ent_emb).tanh(), rel_emb.t().tanh())  # [q, C_way]

    def forward(self, task, meta_support_left, meta_support_right, meta_query_left, meta_query_right):

        support_neighbor_encoding = self.meta_support_encode(meta_support_left,
                                                             meta_support_right, task)  # [C_way, K_shot,2* emb_dim]

        # confidence_left = support_neighbor_encoding[2]
        # confidence_right = support_neighbor_encoding[3]  # (C_way, K_shot)

        query_neighbor_encoding = torch.cat(
            (self.neighbor_encoder(meta_query_left[0], meta_query_left[1], task.query_set[:, 0])[0],
             self.neighbor_encoder(meta_query_right[0], meta_query_right[1], task.query_set[:, 1])[0]),
            dim=-1)  # [C_way * query_size, emb_dim*2]

        # support_neighbor_encoding_mean = torch.mean(torch.cat(support_neighbor_encoding[0:2], -1), dim=1)
        support_encoding = torch.cat(support_neighbor_encoding[0:2], -1)  # (C_way, K_shot, 2*emb_dim)

        # baseline_proto = False
        if self.baseline_proto == 'proto':
            support_encoding = self.proto_FC(support_encoding)
            query_neighbor_encoding = self.proto_FC(query_neighbor_encoding)

            support_mean_encoding = support_encoding.mean(1)

            use_distance = True

            if not use_distance:
                scores = torch.matmul(
                    support_mean_encoding / (support_mean_encoding.norm(p=2, dim=-1, keepdim=True) + 1e-9),
                    (query_neighbor_encoding / (
                            query_neighbor_encoding.norm(p=2, dim=-1, keepdim=True) + 1e-9)).t()).t()
            else:
                def euclidean_dist(x, y):
                    """
                    Args:
                    x: pytorch Variable, with shape [m, d]
                    y: pytorch Variable, with shape [n, d]
                    Returns:
                    dist: pytorch Variable, with shape [m, n]
                    """
                    m, n = x.size(0), y.size(0)
                    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                    dist = xx + yy
                    dist.addmm_(1, -2, x, y.t())
                    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
                    return dist

                scores = -euclidean_dist(query_neighbor_encoding, support_mean_encoding)

            loss = F.cross_entropy(scores, task.query_label)

            # print(scores)
            return scores, loss
        elif self.baseline_proto == 'relation':
            support_mean_encoding = support_encoding.mean(1)  # [N,e]

            concated_emb = torch.cat([support_mean_encoding.unsqueeze(0).repeat(query_neighbor_encoding.shape[0],1, 1),
                                      query_neighbor_encoding.unsqueeze(1).repeat(1,support_mean_encoding.shape[0], 1)],-1)
            scores=self.relation_FC2(self.relation_FC1(concated_emb).relu()).sigmoid().squeeze()
            #loss=torch.pow(scores-torch.zeros(scores.shape).cuda().scatter_(1, task.query_label.unsqueeze(-1), 1),2).sum()
            loss = F.cross_entropy(scores, task.query_label)
            return scores,loss
        #########################
        use_uncertain = True
        use_simi = False
        use_abs = False

        if use_uncertain:

            if use_simi:

                support_encoding_L = support_encoding.reshape([-1, self.L, 2 * self.embed_dim // self.L]).transpose(0,
                                                                                                                    1)  # [L, NK, 2*emb_dim//L]
                query_neighbor_encoding_L = query_neighbor_encoding.reshape(
                    [-1, self.L, 2 * self.embed_dim // self.L]).transpose(0, 1)  # [L, NQ, 2*emb_dim//L]


                V_input = torch.matmul(support_encoding_L, (
                    query_neighbor_encoding_L).transpose(1, 2))  # [L,NK,NQ]
                V_input = V_input.transpose(0, 2)  # [NQ, NK, L]

                E = torch.matmul(self.phi_1(V_input), self.phi_2(V_input).transpose(1, 2))  # [NQ, NK,NK]

                E_normalized = E.softmax(dim=-1)
                V = V_input + self.W_y(self.W_v(E_normalized.matmul(V_input)))  # [NQ, NK, L]

                u = self.W_u(V)  # [NQ, NK,1]
            else:
                support_encoding_L = support_encoding.reshape([-1, 2 * self.embed_dim])  # [NK, 2e]
                query_encoding_L = query_neighbor_encoding  # [NQ, 2e]

                support_encoding_expand = support_encoding_L.unsqueeze(0).repeat(
                    [query_encoding_L.shape[0], 1, 1])  # [NQ, NK, 2e]
                query_encoding_expand = query_encoding_L.unsqueeze(1).repeat(
                    [1, support_encoding_L.shape[0], 1])  # [NQ,NK,2e]

                mix = torch.cat(
                    [support_encoding_expand, query_encoding_expand, support_encoding_expand + query_encoding_expand,
                     support_encoding_expand * query_encoding_expand], -1)  # [NQ, NK, 8e]

                V_input = self.G_w(mix)  # [NQ, NK, e]

                if not use_abs:
                    E = torch.matmul(self.p_1(V_input), V_input.transpose(1, 2))  # [NQ, NK,NK]
                else:
                    features = V_input.reshape(-1, self.embed_dim)
                    fea_num = features.shape[0]
                    features_0 = features.repeat(fea_num, 1)  # [n*n, H]
                    features_1 = features.repeat(1, fea_num).reshape(fea_num * fea_num, -1)  # [n*n,H]

                    E = self.W_abs(torch.abs(features_0 - features_1)).reshape(
                        [V_input.shape[0], V_input.shape[1], V_input.shape[1]])

                E_normalized = E.softmax(dim=-1)

                V = V_input + torch.nn.functional.leaky_relu(self.G_y(E_normalized.matmul(V_input)), 0.2)  # [NQ, NK, L]

                u = self.G_u(V)  # [NQ, NK,1]

            att = False

            if att:
                query_neighbor_encoding = u.squeeze().unsqueeze(1).softmax(-1).matmul(
                    support_encoding.reshape([1, task.C_way * self.K_shot, -1]).repeat([u.shape[0], 1, 1])).squeeze()

            mean = False

            if mean:
                u = u.reshape([u.shape[0], task.C_way, self.K_shot]).mean(dim=2, keepdim=False)  # [NQ, N]

                self.u = u[0]
            else:
                v = u.reshape([u.shape[0], task.C_way, self.K_shot]).max(dim=2, keepdim=False)[0]
                u = v.sigmoid()  # [NQ, N]

                self.u = u[0]
                self.v_total = v

        else:
            self.u = 0

        #########################

        use_cross_attention = True

        r_hat = self.transformer_instance(support_encoding.transpose(1, 0)).transpose(1,
                                                                                      0)  # (C_way, K_shot, 2*emb_dim)
        r_hat = r_hat.mean(1)  # (C_way, 2*emb_dim)

        r_att = self.transformer_relations(r_hat.unsqueeze(0).transpose(1, 0)).transpose(1,
                                                                                         0).squeeze()  # (C_way, 2*emb_dim)
        r_att = r_att.mean(0)

        if use_cross_attention:
            r_final = r_hat * r_att  # (C_way, 2*emb_dim)
        else:
            r_final = r_hat

        r_final_split = r_final.split([r_final.shape[0] // 2, r_final.shape[0] - r_final.shape[0] // 2])



        scores = -self.energy_func(query_neighbor_encoding, r_final, if_query=True)  # (C_way * query_size, C_way)

        if use_uncertain:

            if not att:


                scores = scores.softmax(-1) * (u + 1e-9)
                scores /= scores.sum(dim=-1, keepdim=True)
                scores = scores.log()
                loss_f = nn.NLLLoss()
                loss = loss_f(scores, task.query_label)



            else:
                loss = F.cross_entropy(scores, task.query_label)



        else:
            loss = F.cross_entropy(scores, task.query_label)

        return scores, loss


result = {}
rel_acc = {}


for dataset in ['FB15K_0.0_t33']:
#for dataset in ['NELL', 'NELL_0.05', 'NELL_0.1', 'NELL_0.15', 'NELL_0.2']:
#for dataset in ['NELL']:
#for dataset in ['Wiki', 'Wiki_0.05','Wiki_0.1','Wiki_0.15', 'Wiki_0.2']:
    batch_max = 5000
    for few_shot in [5]:
        for i in range(1):
            save_rel = True
            trainer = Trainer('./' + dataset, 'TransE', num_class, few_shot, './save_model', max_neighbor=50,
                              query_size=5, batch_num_max=batch_max, save_rel=save_rel, baseline_proto='relation')
            score = trainer.train()
            if dataset not in result:
                result[dataset] = [score]
            else:
                result[dataset].append(score)
            if save_rel:
                if dataset not in rel_acc:
                    rel_acc[dataset] = [[trainer.rel_dev_best_acc, trainer.rel_test_best_acc]]
                else:
                    rel_acc[dataset].append([trainer.rel_dev_best_acc, trainer.rel_test_best_acc])
                json.dump(rel_acc, open('./rel_acc_Wiki_relation.json', 'w'))
            del trainer
        json.dump(result, open('./result_acc_Wiki_relation.json', 'w'))

