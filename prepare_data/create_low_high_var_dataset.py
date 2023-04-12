import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
cos=torch.nn.CosineSimilarity()
mtcnn = MTCNN(image_size=160)
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from k_means_constrained import KMeansConstrained
from sklearn.metrics.pairwise import cosine_similarity

# load data
# list of paths --> load each one -> preprocess
def load_data(data_dir,class_num_start,class_num_end, instance_num):
    data_paths_list=[]
    for c in os.listdir(data_dir)[class_num_start:class_num_end]:
        class_path = os.path.join(data_dir, c)
        i=0
        while i<instance_num+1:
            random_image = random.choice(os.listdir(class_path))
            im_path = os.path.join(class_path,random_image)
            data_paths_list.append(im_path)
            i+=1

    return data_paths_list



## load model
def load_model():
    resnet = InceptionResnetV1(pretrained='casia-webface').eval()
    return resnet


def get_embeddings(dataset_paths_list, model):
    transform_totensor = transforms.Compose([transforms.PILToTensor()])
    embeddings_list = np.zeros((len(dataset_paths_list), 512))
    labels=np.zeros(len(dataset_paths_list))
    for i,im_path in enumerate(dataset_paths_list):
        try:
            img = Image.open(im_path)
            img=mtcnn(img)
            #img = transform_totensor(img)
            img_embedding = model(img.unsqueeze(0).float())
            embeddings_list[i]= img_embedding.detach().numpy()
            labels[i]=im_path.split("train")[-1].split('.')[0][3:8]

        except:
            print (im_path)
            labels[i]=1744

    return embeddings_list, labels

def get_TSNE(X):
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    return tsne_result

def get_closest_x_clusters(tsne_results, cluster_num, x_target):
    cluster_means=[]
    for i in range(cluster_num):
        cluster_values = tsne_results[(tsne_results['cluster'] == i)][['x1','x2']]
        cluster_means.append(np.mean(cluster_values, axis=0))
    rdm = np.zeros((len(cluster_means), len(cluster_means)))
    for i, mean1 in enumerate(cluster_means):
        for j, mean2 in enumerate(cluster_means):
            x_y = mean1.x1 * mean2.x1 + mean1.x2 * mean2.x2
            x_norm = np.sqrt((mean1.x1**2)+(mean1.x2**2))
            y_norm = np.sqrt((mean2.x1**2)+(mean2.x2**2))
            rdm[i, j] = x_y/(x_norm*y_norm)
    plt.imshow(rdm)
    plt.show()
    cluster_sim = np.sum(rdm, axis=0)
    most_dissimilar_cluster = np.min(cluster_sim)
    return cluster_means


def create_dissimilar_training_from_clusters(tsne_results_df, size, num_clusters):
    new_set = []
    num_instances_per_cluster = np.ceil(size / num_clusters).astype(int)
    for cluster in range(num_clusters):
        try:
            curr_cluster_idx = tsne_results_df[tsne_results_df['cluster'] == cluster].index.values.astype(int)
            new_set.extend(np.random.choice(curr_cluster_idx, num_instances_per_cluster, replace=False))
        except:
            num_instances_per_cluster+=1
            continue
    if len(new_set) > size:
        new_set = new_set[0:size + 1]
    return new_set


def get_tsne(embeddings_list, n_clusters, size_min):
    tsne_result = get_TSNE(embeddings_list)
    tsne_result_df = pd.DataFrame(tsne_result)
    clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min)
    clustering_ori = clf.fit_predict(tsne_result_df)
    tsne_result_df['cluster'] = clustering_ori
    tsne_result_df.columns = ['x1', 'x2', 'cluster']
    return tsne_result_df


if __name__ == '__main__':
    cropped_align_dir = r'C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\data\train'
    root_dir = r"C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\high_low_var_splits"
    low_var_path = os.path.join(root_dir, 'low_var')
    high_var_path = os.path.join(root_dir, 'high_var')
    low_var_train_path = os.path.join(low_var_path, 'training')
    low_var_similar_path = os.path.join(low_var_path, 'similar')
    low_var_dissimilar_path = os.path.join(low_var_path, 'dissimilar')
    high_var_train_path = os.path.join(high_var_path, 'training')
    high_var_similar_path = os.path.join(high_var_path, 'similar')
    high_var_dissimilar_path = os.path.join(high_var_path, 'dissimilar')
    dirs = [low_var_path, high_var_path, low_var_train_path, low_var_similar_path, low_var_dissimilar_path,
            high_var_train_path, high_var_similar_path, high_var_dissimilar_path]
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)

    NUM_CLASSES= 20
    INSTANCE_NUM=200
    MIN_SIZE_CLASS = 60
    TRAINING_SET_SIZE=50
    TEST_SET_SIZE = MIN_SIZE_CLASS - TRAINING_SET_SIZE

    for c in tqdm(range(NUM_CLASSES)):
        data_paths_list = load_data(cropped_align_dir, class_num_start=c, class_num_end=c+1, instance_num=INSTANCE_NUM)
        if len(data_paths_list) <INSTANCE_NUM:
            print(f'class does not contain enough images, {data_paths_list[0]}')
            continue
        model = load_model()
        embeddings_list, labels = get_embeddings(data_paths_list, model)

        #low var training
        CLUSTER_NUMBER = 3
        tsne_result = get_TSNE(embeddings_list)
        tsne_result_df = pd.DataFrame(tsne_result)
        clf = KMeansConstrained(n_clusters=CLUSTER_NUMBER,size_min=MIN_SIZE_CLASS)
        clustering_ori = clf.fit_predict(tsne_result_df)
        tsne_result_df['cluster'] = clustering_ori
        tsne_result_df.columns = ['x1', 'x2', 'cluster']
        first_cluster_idx = tsne_result_df[tsne_result_df['cluster'] == 0].index.values.astype(int)

        for i, j in enumerate(first_cluster_idx[:TRAINING_SET_SIZE+1]):
            im_path = data_paths_list[j]
            img = Image.open(im_path)
            class_name= im_path.split("train")[-1].split('.')[0][3:8]
            class_path = os.path.join(low_var_train_path, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            img.save(f'{class_path}/{i}.jpg')

        #similar test
        for i, j in enumerate(first_cluster_idx[TRAINING_SET_SIZE:]):
            im_path = data_paths_list[j]
            img = Image.open(im_path)
            class_name= im_path.split("train")[-1].split('.')[0][3:8]
            class_path = os.path.join(low_var_similar_path, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            img.save(f'{class_path}/{i}.jpg')

        #dissimilar test
        other_clusters_idx = tsne_result_df[tsne_result_df['cluster'] != 0].index.values.astype(int)
        dissimilar = np.random.choice(other_clusters_idx, TEST_SET_SIZE, replace= False)
        for i, j in enumerate(dissimilar):
            im_path = data_paths_list[j]
            img = Image.open(im_path)
            class_name= im_path.split("train")[-1].split('.')[0][3:8]
            class_path = os.path.join(low_var_dissimilar_path, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            img.save(f'{class_path}/{i}.jpg')


        # high var training
        # divide into 2 clusters - one will be used for training and similar, other for dissimilar
        CLUSTER_NUMBER = 2
        HALF_SIZE_CLASS = MIN_SIZE_CLASS//2
        tsne_result_df_high_var = get_tsne(embeddings_list, CLUSTER_NUMBER, HALF_SIZE_CLASS)
        high_var_training_idxs= tsne_result_df_high_var[tsne_result_df_high_var['cluster'] == 0].index.values.astype(int)
        high_var_dissimilar_idxs= tsne_result_df_high_var[tsne_result_df_high_var['cluster'] == 1].index.values.astype(int)


        #training
        CLUSTER_NUMBER = 10
        MIN_SIZE_CLASS = TRAINING_SET_SIZE//CLUSTER_NUMBER
        high_var_training_datapaths_list = [data_paths_list[i] for i in high_var_training_idxs]
        new_embeddings_list, new_labels = get_embeddings(high_var_training_datapaths_list, model)
        tsne_result_df_high_var_training = get_tsne(new_embeddings_list, CLUSTER_NUMBER, MIN_SIZE_CLASS)

        high_var_training_set = create_dissimilar_training_from_clusters(tsne_result_df_high_var_training, TRAINING_SET_SIZE+TEST_SET_SIZE, CLUSTER_NUMBER)
        for i, j in enumerate(high_var_training_set[: TRAINING_SET_SIZE]):
            im_path =  high_var_training_datapaths_list[j]
            img = Image.open(im_path)
            class_name= im_path.split("train")[-1].split('.')[0][3:8]
            class_path = os.path.join(high_var_train_path, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            img.save(f'{class_path}/{i}.jpg')

        #similar
        for i, j in enumerate(high_var_training_set[TRAINING_SET_SIZE: ]):
            im_path = high_var_training_datapaths_list[j]
            img = Image.open(im_path)
            class_name= im_path.split("train")[-1].split('.')[0][3:8]
            class_path = os.path.join(high_var_similar_path, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            img.save(f'{class_path}/{i}.jpg')


        #dissimilar
        high_var_dissimilar_datapaths_list = [data_paths_list[i] for i in high_var_dissimilar_idxs]
        dissimilar = np.random.choice(high_var_dissimilar_datapaths_list, TEST_SET_SIZE, replace=False)
        for i, j in enumerate(dissimilar):
            im_path =  j
            img = Image.open(im_path)
            class_name= im_path.split("train")[-1].split('.')[0][3:8]
            class_path = os.path.join(high_var_dissimilar_path, class_name)
            if not os.path.isdir(class_path):
                os.mkdir(class_path)
            img.save(f'{class_path}/{i}.jpg')
