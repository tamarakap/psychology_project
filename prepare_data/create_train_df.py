import os
import pandas as pd
from tqdm import tqdm
from training.config import TrainConfig
from sklearn.model_selection import train_test_split


def create_unique_train_set_df(config):
    '''
    This function creates a csv file with a unique train set, with the classes num and instances num from config.
    Each row in the df will contain the class, instance num, file name, full path, and set the image belongs to.
    This file is saved in the general dfs folder, so it can be used for another experiments with the same conditions.
    '''
    data_dir = config.train_data_dir
    num_classes = config.num_classes
    train_size = config.train_num_instances_per_class
    val_size = config.val_num_instances_per_class

    df = pd.DataFrame(columns=[['class', 'instance_num', 'image_name', 'image_path','set']])
    classes = os.listdir(data_dir)
    for class_num, cl in tqdm(enumerate(classes[0: num_classes])):
        class_df = pd.DataFrame(columns=df.columns)
        class_path = os.path.join(data_dir, cl)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_set, val_set = train_test_split(images, test_size=val_size, train_size=train_size, random_state=42)
            for set, name in zip([train_set, val_set], ['train','val']):
                for img in set:
                    img_path = os.path.join(class_path, img)
                    row = [class_num, len(class_df), img, img_path, name]
                    class_df.loc[len(class_df)]=row
        df = pd.concat([df, class_df], axis =0)
    return df

if __name__ == '__main__':
    config = TrainConfig()
    num_classes = config.num_classes
    num_instances_per_class = config.train_num_instances_per_class
    df_dir = f'{config.root_dir}/training_set_dfs'
    df_path = f'{df_dir}/{num_classes}_classes_{num_instances_per_class}_per_class.csv'
    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)
    else:
        df = create_unique_train_set_df(config)
        df.to_csv(df_path)