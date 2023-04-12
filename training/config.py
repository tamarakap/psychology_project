import os
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    root_dir: str = r'C:\Users\shiri\Documents\School\faces'
    train_data_dir: str =f'{root_dir}/Data/VGG-Face2/data/train_mtcnn_aligned'
    results_dir: str =f'{root_dir}/vgg_exps'


    make_new_train_set: bool = False
    num_classes: int = 20
    train_num_instances_per_class: int =50
    val_num_instances_per_class: int = 30

    epochs:int = 10
    batch_size: int = 4

    cos_thresh=0.7

    # data pre processing
    align_data = False
    apply_mtcnn = False
    model_type = 'resnet'
    rescale_size = (160,160)

    # Run cosine similarity test every epoch - optional
    run_tester: bool = False
    df_test: str = r"C:\Users\shiri\Documents\School\Galit\Data\LFW\lfw-py\lfw_funneled\test_df2.csv"