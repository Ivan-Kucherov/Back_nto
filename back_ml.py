import pickle
import numpy as np
from geopy.distance import geodesic as GD
import torch
import umap
import os


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def cosine_similarity(x, y):

    # Ensure length of x and y are the same
    if len(x) != len(y) :
        return None

    # Compute the dot product between x and y
    dot_product = np.dot(x, y)

    # Compute the L2 norms (magnitudes) of x and y
    magnitude_x = np.sqrt(np.sum(x**2))
    magnitude_y = np.sqrt(np.sum(y**2))

    # Compute the cosine similarity
    cosine_similarity = dot_product / (magnitude_x * magnitude_y)

    return cosine_similarity

class model:
    def __init__(self, path_kmeans):
        self.path_kmeans = path_kmeans
    
    def __new__(self):
        self.kmeans =  pickle.load(open('save.pkl','rb'))


    def get_embeddings(self, images):
        device = 'cuda'
        clip, processor = ruclip.load("ruclip-vit-base-patch32-384", device="cuda")
        predictor = ruclip.Predictor(clip, processor, device, bs=8)
        for i in range(len(images)):
            with torch.no_grad():
                emb = predictor.get_image_latents([images[i]['img']])
            images[i]['img'] = self.dimension_model.transform(emb.cpu().detach().numpy())
        return images
        

    '''images_ = [{'img':test_clusterable_embedding[a],'pos':(X_test.iloc[a]['lat'], X_test.iloc[a]['lon'])}]'''
    def get_preds(self, images_):
        kmeans, X_train, train_clusterable_embedding = self.kmeans, self.X_train, self.train_clusterable_embedding

        images_ = self.get_embeddings(images_)

        clusters = {}
        for i in range(len(images_)):
            if kmeans.predict(np.expand_dims(images_[i]['img'], axis=0))[0] in clusters.keys():
                clusters[kmeans.predict(np.expand_dims(images_[i]['img'], axis=0))[0]].append(i)
            else:
                clusters[kmeans.predict(np.expand_dims(images_[i]['img'], axis=0))[0]] = [i]

        concat = []
        indexes = []
        for key,val in clusters.items():
            #print(key,val)
            temp_df = X_train[X_train['number_of_cluster'] == key]
            ind = temp_df.index
            vec = np.zeros(len(temp_df))
            for i in val:
                vec += np.array([cosine_similarity(images_[i]['img'], train_clusterable_embedding[j]) for j in temp_df.index])
            clusters[key] = {'value': normalize(vec, 0, 1),'index':ind}
            concat += clusters[key]['value']
            indexes += ind.to_list()

        ind = []
        for i in sorted(concat, reverse=True):
            if len(ind)>=5:
                break
            cnt = False
            for j in images_:
                if (GD((X_train.iloc[indexes[concat.index(i)]]['lat'], X_train.iloc[indexes[concat.index(i)]]['lon']), j['pos']).m < 150):
                    cnt =True
            for j in ind:
                if (GD((X_train.iloc[indexes[concat.index(i)]]['lat'], X_train.iloc[indexes[concat.index(i)]]['lon']), (X_train.iloc[j]['lat'], X_train.iloc[j]['lon'])).m < 50):
                    cnt =True
            if not(cnt):
                ind.append(indexes[concat.index(i)])
        return [(X_train.iloc[i]['img'],( X_train.iloc[i]['lon'], X_train.iloc[i]['lat'])) for i in ind]
    
    def get_nearest(self, images_):
        kmeans, X_train, train_clusterable_embedding = self.kmeans, self.X_train, self.train_clusterable_embedding
        images_ = self.get_embeddings(images_)

        clusters = {}
        for i in range(len(images_)):
            if kmeans.predict(np.expand_dims(images_[i]['img'], axis=0))[0] in clusters.keys():
                clusters[kmeans.predict(np.expand_dims(images_[i]['img'], axis=0))[0]].append(i)
            else:
                clusters[kmeans.predict(np.expand_dims(images_[i]['img'], axis=0))[0]] = [i]

        concat = []
        indexes = []
        for key,val in clusters.items():
            #print(key,val)
            temp_df = X_train[X_train['number_of_cluster'] == key]
            ind = temp_df.index
            concat_ = []
            for i in val:
                concat_ += [GD(images_[-1]['pos'], (temp_df.loc[j]['lat'],temp_df.loc[j]['lon'] )).m for j in temp_df.index]

            clusters[key] = {'value': concat_,'index':ind}
            concat += clusters[key]['value']
            indexes += ind.to_list()


        ind = []
        d = []
        for i in sorted(concat, reverse=False):
            if len(ind)>=5:
                break
            cnt = False
            for j in images_:
                if (GD((X_train.iloc[indexes[concat.index(i)]]['lat'], X_train.iloc[indexes[concat.index(i)]]['lon']), j['pos']).m < 150):
                    cnt =True
            for j in ind:
                if (GD((X_train.iloc[indexes[concat.index(i)]]['lat'], X_train.iloc[indexes[concat.index(i)]]['lon']), (X_train.iloc[j]['lat'], X_train.iloc[j]['lon'])).m < 50):
                    cnt =True
            if not(cnt):
                ind.append(indexes[concat.index(i)])
                d.append(i)
        return ([(X_train.iloc[i]['img'], X_train.iloc[i]['lon'], X_train.iloc[i]['lat']) for i in ind], d)