import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


for dataset in ['Wiki_0.15','Wiki_0.2']:


    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path = "../{}/".format(dataset),
        nbatches = 100,
        threads = 8, 
        sampling_mode = "normal", 
        bern_flag = 1, 
        filter_flag = 1, 
        neg_ent = 25,
        neg_rel = 0)

    # dataloader for test
    #test_dataloader = TestDataLoader("./drug_KG/", "link")

    # define the model
    transe = TransE(
        ent_tot = train_dataloader.get_ent_tot(),
        rel_tot = train_dataloader.get_rel_tot(),
        dim = 50, 
        p_norm = 1, 
        norm_flag = True)

    #transe.load_checkpoint('./transe.ckpt')


    # define the loss function
    model = NegativeSampling(
        model = transe, 
        loss = MarginLoss(margin = 5.0),
        batch_size = train_dataloader.get_batch_size()
    )

    # train the model

    trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 2000, alpha = 1.0, use_gpu = True)
    trainer.run()
    transe.save_checkpoint('../{}/transe.ckpt'.format(dataset))
    transe.save_parameters('../{}/transe.json'.format(dataset))

    import pickle as pkl
    import numpy as np
    import json

    #emb=json.load(open('../{}/transe.json'.format(dataset)))


    #np.savetxt(open('../{}/entity2vec.TransE'.format(dataset),'w'),np.array(emb['ent_embeddings.weight']))
    #np.savetxt(open('../{}/relation2vec.TransE'.format(dataset),'w'),np.array(emb['rel_embeddings.weight']))

    # test the model
    #transe.load_checkpoint('./checkpoint/transe.ckpt')
    #tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
    #tester.run_link_prediction(type_constrain = False)