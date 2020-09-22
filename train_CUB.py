from torch.autograd import Variable
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import argparse
import os
import random
import json
from dataset import DatasetCUB
import copy
import scipy.integrate as integrate


from torch.utils import data

import torch
from models import Attention


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--split', default='easy', type=str, help='the way to split train/test data: easy/hard')
parser.add_argument('--model', default='similarity_VRS', type=str, help='the type of ZEST model to use: vanilla/similarity/similarity_VRS')
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--max_epoch',  type=int, default=500)


opt = parser.parse_args()
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ':')))

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


if opt.seed is None:
    opt.seed = random.randint(1, 10000)
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


opt.lr = 0.0005
opt.batch_size=100


def train():
    params = {'batch_size': opt.batch_size, 
              'shuffle': True,
              'num_workers': 0}

    torch.backends.cudnn.benchmark = True
    training_set = DatasetCUB(opt)
    training_generator = data.DataLoader(training_set, **params)
    test_set = DatasetCUB(opt,train=False)
    test_generator = data.DataLoader(test_set, **params)
    netA=Attention(text_dim=training_set.text_dim, dimensions=training_set.feature_dim).cuda()
    netA.apply(weights_init)
    optimizerA = optim.Adam(netA.parameters(), lr=opt.lr, betas=(0.5, 0.9), weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    text_feat=Variable(torch.tensor(training_set.train_text_feature)).unsqueeze(0).cuda()
    text_feat_test=Variable(torch.tensor(training_set.test_text_feature)).unsqueeze(0).cuda()

    for it in range(opt.max_epoch):
        print('epoch: ', it)

        for bi, batch in enumerate(training_generator):

            images, labels = batch
            image_representation, y_true = Variable(images).cuda(), labels.cuda()

            attention_weights,attention_scores=netA(image_representation,text_feat)

            loss = criterion(attention_weights.squeeze(), y_true.long())

            topv, topi = attention_scores.squeeze().data.topk(1)
            compare_pred_ground = topi.squeeze() == y_true
            correct = np.count_nonzero(compare_pred_ground.cpu() == 1)

            optimizerA.zero_grad()
            loss.backward()
            optimizerA.step()

        # print("it:", it)

        # print('train accuracy:', correct / y_true.shape[0])
        netA.eval()

        correct=0

        for bi, batch in enumerate(test_generator):

            images, labels = batch

            image_representation, y_true = Variable(images).cuda(), labels.cuda()
            attention_weights, attention_scores = netA(image_representation, text_feat_test)
            topv, topi = attention_weights.squeeze().data.topk(1)
            correct+=torch.sum(topi.squeeze()==y_true).cpu().tolist()

        print (test_set.pfc_feat_data_test.shape)
        print('test accuracy:', 100 * correct / test_set.pfc_feat_data_test.shape[0])
        GZSL_evaluation(text_feat, text_feat_test,training_set.train_cls_num,training_generator,test_generator,netA)
        netA.train()

def GZSL_evaluation(text_feat_train, text_feat_test,train_cls_num,training_generator,test_generator,netA):

    text_feat=torch.cat((text_feat_train,text_feat_test),dim=1)

    # unseen
    unseen_sim = np.zeros([0, text_feat.shape[1]])
    labels_unseen = []
    for bi, batch in enumerate(test_generator):
        images, labels = batch
        labels += train_cls_num
        labels_unseen += labels.tolist()
        image_representation, y_true = Variable(images).cuda(), labels.cuda()

        attention_weights, attention_scores = netA(image_representation, text_feat)
        unseen_sim = np.vstack((unseen_sim, attention_weights.squeeze().data.cpu().numpy()))

    seen_sim = np.zeros([0, text_feat.shape[1]])
    labels_seen = []
    for bi, batch in enumerate(training_generator):
        images, labels = batch
        image_representation, y_true = Variable(images).cuda(), labels.cuda()
        labels_seen += labels.tolist()

        attention_weights, attention_scores = netA(image_representation, text_feat)

        seen_sim = np.vstack((seen_sim, attention_weights.squeeze().data.cpu().numpy()))

    acc_S_T_list, acc_U_T_list = list(), list()

    for GZSL_lambda in np.arange(-2, 2, 0.01):
        tmp_seen_sim = copy.deepcopy(seen_sim)
        tmp_seen_sim[:, train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_seen_sim, axis=1)
        acc_S_T_list.append((pred_lbl == np.asarray(labels_seen)).mean())

        tmp_unseen_sim = copy.deepcopy(unseen_sim)
        tmp_unseen_sim[:, train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_unseen_sim, axis=1)
        acc_U_T_list.append((pred_lbl == (np.asarray(labels_unseen))).mean())

    auc_score = integrate.trapz(y=acc_S_T_list, x=acc_U_T_list)

    print("AUC Score is {:.4}".format(auc_score))



def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)


if __name__ == "__main__":
    train()