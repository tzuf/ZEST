import numpy as np
import scipy.io as sio
from termcolor import cprint
import pickle
import sys
import os
from torch.utils.data import Dataset
import torch
from sklearn.cluster import DBSCAN
from sklearn import metrics
import hdbscan

class DatasetNAB(Dataset):
    def __init__(self, opt, train=True):
        self.train = train
        txt_feat_path_original = r"data/NABird/NAB_Porter_13217D_TFIDF_new_original.mat"

        if opt.split == 'easy':
            if opt.model=='similarity_VRS': #'similarity_VRS'
                txt_feat_path="data/NABird/NAB_EASY_SPLIT_VRS.mat"
            else: #vanilla\similarity
                txt_feat_path = r"data/NABird/NAB_Porter_13217D_TFIDF_new_original.mat"

            train_test_split_dir = 'data/NABird/train_test_split_NABird_easy.mat'
            pfc_label_path_train = 'data/NABird/labels_train.pkl'
            pfc_label_path_test = 'data/NABird/labels_test.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_easy.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_easy.mat'
            train_cls_num = 323
            test_cls_num = 81
        else:
            if opt.model=='similarity_VRS': #'similarity_VRS'
                txt_feat_path="data/NABird/NAB_HARD_SPLIT_VRS.mat"
            else: #vanilla\similarity
                txt_feat_path = r"data/NABird/NAB_Porter_13217D_TFIDF_new_original.mat"


            train_test_split_dir = 'data/NABird/train_test_split_NABird_hard.mat'
            pfc_label_path_train = 'data/NABird/labels_train_hard.pkl'
            pfc_label_path_test = 'data/NABird/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/NABird/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/NABird/pfc_feat_test_hard.mat'
            train_cls_num = 323
            test_cls_num = 81

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)

        self.train_cls_num = train_cls_num
        self.test_cls_num = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]
        # calculate the corresponding centroid.
        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        if self.train==True:
            similarity_flag= 0 if opt.model=='vanilla' else 1
            self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir,txt_feat_path_original,similarity_flag)
            self.text_dim = self.train_text_feature.shape[1]


    def __len__(self):
        if self.train:
            return len(self.pfc_feat_data_train)
        else:
            return len(self.pfc_feat_data_test)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if self.train == True:
            I = torch.from_numpy(self.pfc_feat_data_train[index])
            label = torch.tensor(self.labels_train[index],dtype=torch.float64)
            return I, label
        else:
            I = torch.from_numpy(self.pfc_feat_data_test[index])
            label = torch.tensor(self.labels_test[index],dtype=torch.float64)

            return I, label

class DatasetCUB(Dataset):
    def __init__(self, opt, train=True):
        self.train = train

        txt_feat_path_original = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"

        if opt.split == 'easy':
            if opt.model=='similarity_VRS': #'similarity_VRS'
                txt_feat_path=r"data/CUB2011/CUB_EASY_SPLIT_VRS.mat"
            else: #vanilla\similarity
                txt_feat_path = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"

            train_test_split_dir = 'data/CUB2011/train_test_split_easy.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test.mat'
            train_cls_num = 150
            test_cls_num = 50
        else:
            if opt.model=='similarity_VRS': #'similarity_VRS'
                txt_feat_path=r"data/CUB2011/CUB_HARD_SPLIT_VRS.mat"
            else: #vanilla\similarity
                txt_feat_path = r"data/CUB2011/CUB_Porter_7551D_TFIDF_new_original.mat"

            train_test_split_dir = 'data/CUB2011/train_test_split_hard.mat'
            pfc_label_path_train = 'data/CUB2011/labels_train_hard.pkl'
            pfc_label_path_test = 'data/CUB2011/labels_test_hard.pkl'
            pfc_feat_path_train = 'data/CUB2011/pfc_feat_train_hard.mat'
            pfc_feat_path_test = 'data/CUB2011/pfc_feat_test_hard.mat'
            train_cls_num = 160
            test_cls_num = 40

        self.pfc_feat_data_train = sio.loadmat(pfc_feat_path_train)['pfc_feat'].astype(np.float32)
        self.pfc_feat_data_test = sio.loadmat(pfc_feat_path_test)['pfc_feat'].astype(np.float32)

        self.train_cls_num = train_cls_num
        self.test_cls_num = test_cls_num
        self.feature_dim = self.pfc_feat_data_train.shape[1]

        with open(pfc_label_path_train, 'rb') as fout1, open(pfc_label_path_test, 'rb') as fout2:
            if sys.version_info >= (3, 0):
                self.labels_train = pickle.load(fout1, encoding='latin1')
                self.labels_test = pickle.load(fout2, encoding='latin1')
            else:
                self.labels_train = pickle.load(fout1)
                self.labels_test = pickle.load(fout2)

        # Normalize feat_data to zero-centered
        mean = self.pfc_feat_data_train.mean()
        var = self.pfc_feat_data_train.var()
        self.pfc_feat_data_train = (self.pfc_feat_data_train - mean) / var
        self.pfc_feat_data_test = (self.pfc_feat_data_test - mean) / var

        if self.train==True:
            similarity_flag= 0 if opt.model=='vanilla' else 1
            self.train_text_feature, self.test_text_feature = get_text_feature(txt_feat_path, train_test_split_dir,txt_feat_path_original,similarity_flag)
            self.text_dim = self.train_text_feature.shape[1]


    def __len__(self):
        if self.train:
            return len(self.pfc_feat_data_train)
        else:
            return len(self.pfc_feat_data_test)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if self.train == True:
            I = torch.from_numpy(self.pfc_feat_data_train[index])
            label = torch.tensor(self.labels_train[index],dtype=torch.float64)
            return I, label
        else:
            I = torch.from_numpy(self.pfc_feat_data_test[index])
            label = torch.tensor(self.labels_test[index],dtype=torch.float64)

            return I, label


def get_text_feature(dir, train_test_split_dir,txt_feat_path_original, add_similarity_flag=True):
    train_test_split = sio.loadmat(train_test_split_dir)
    train_cid = train_test_split['train_cid'].squeeze()-1
    test_cid = train_test_split['test_cid'].squeeze()-1

    text_feature = sio.loadmat(dir)['PredicateMatrix']
    text_feature_original=sio.loadmat(txt_feat_path_original)['PredicateMatrix']

    text_feature_new, intersections=add_similarity(text_feature,train_cid,test_cid,text_feature_original)
    if intersections.shape[0]/test_cid.shape[0]>0.15 and add_similarity_flag:
        print ('added similarity features')
        text_feature=text_feature_new

    train_text_feature = text_feature[train_cid]  # 0-based index
    test_text_feature = text_feature[test_cid]  # 0-based index

    return train_text_feature.astype(np.float32), test_text_feature.astype(np.float32)


def family_addition_features(text_features, path_text):
    files = os.listdir(path_text)
    files_xls = [f for f in files if f[-3:] == 'txt']
    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    dic_family = {}
    list_family = []
    for counter, file_name in enumerate(files_xls):
        family = file_name.split("_")[-1]
        if family not in dic_family:
            dic_family[family] = len(dic_family)
        list_family.append(dic_family[family])


    family_fetures=np.zeros((len(list_family),len(dic_family)))
    for l_i,label in enumerate(list_family):
        zero_current=np.zeros(len(dic_family))
        zero_current[label]=1
        family_fetures[l_i]=zero_current


    return np.concatenate((family_fetures,text_features),axis=1)



def evaluate_cluster(clustering, path_dir_text):
    from sklearn.metrics import accuracy_score
    import operator

    cluster_labels=[]
    counter_max_label=max(clustering.labels_)+1
    for i in clustering.labels_:
        if i==-1:
            label=counter_max_label
            counter_max_label+=1
        else:
            label=i
        cluster_labels.append(label)

    files = os.listdir(path_dir_text)
    files_xls = [f for f in files if f[-3:] == 'txt']

    files_xls.sort(key=lambda x: int(x.split(".")[0]))

    ground_truth=[]
    prdeiction_label={}
    counter_family=-1
    dic_family={}
    list_family=[]
    for counter,file_name in enumerate(files_xls):
        # print (counter)
        family=file_name.split("_")[-1]
        if family not in dic_family:
            counter_family+=1
            dic_family[family]=counter_family
        list_family.append(dic_family[family])

        if family not in prdeiction_label:
            prdeiction_label[family]={}
        if cluster_labels[counter] not in prdeiction_label[family]:
            prdeiction_label[family][cluster_labels[counter]]=0

        prdeiction_label[family][cluster_labels[counter]] +=1

    for counter,file_name in enumerate(files_xls):
        family=file_name.split("_")[-1]


        cluster_id=max(prdeiction_label[family].items(), key=operator.itemgetter(1))[0]


        ground_truth.append(cluster_id)


    accuracy_score=accuracy_score(cluster_labels,ground_truth)
    print ("cluaster accuracy: ",accuracy_score )
    print ("cluster cluster:", metrics.adjusted_rand_score(list_family, clustering.labels_))
    print ("cluster adjusted_mutual_info_score:",metrics.adjusted_mutual_info_score(list_family, clustering.labels_))



def add_similarity(text_feat,train_id, test_id,original_tetx):
    eps = 0.65
    clustering = hdbscan.HDBSCAN(min_cluster_size=2).fit(original_tetx)

    clustering_2 = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(original_tetx)

    ids_train = clustering.labels_[train_id]
    ids_test = clustering.labels_[test_id]
    (clusters_train, counts_train) = np.unique(ids_train, return_counts=True)
    (cluster_test, counts_test) = np.unique(ids_test, return_counts=True)
    intersections=np.intersect1d(cluster_test, clusters_train)

    n_clusters=np.unique(clustering.labels_, return_counts=False).shape[0]
    n_clusters_2=np.unique(clustering_2.labels_, return_counts=False).shape[0]


    family = np.zeros((text_feat.shape[0], n_clusters))
    family_2 = np.zeros((text_feat.shape[0], n_clusters_2))
    for i in range(text_feat.shape[0]):
        np_zero = np.zeros(n_clusters)
        text_cluster=clustering.labels_[i]
        if text_cluster!=-1:
            np_zero[text_cluster] = 1
        family[i] = np_zero

        np_zero_2 = np.zeros(n_clusters_2)
        text_cluster_2=clustering_2.labels_[i]
        if text_cluster_2!=-1:
            np_zero_2[text_cluster_2] = 1
        family_2[i] = np_zero_2

    text_feat = np.concatenate((family_2,family, text_feat), axis=1)
    return text_feat,intersections

