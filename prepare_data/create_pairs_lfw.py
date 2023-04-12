import os
import pandas as pd


def create_pos_pairs(dir, num_pairs):
    df = pd.DataFrame(columns=[['label', '1', '2']])
    classes = os.listdir(dir)
    for c in classes[40:]:
        class_path = os.path.join(dir, c)
        if os.path.isdir(class_path):
            instances = os.listdir(class_path)
            if len(instances)<=4:
                continue
            else:
                pair1 = [1, os.path.join(class_path,instances[0]),os.path.join(class_path,instances[1])]
                #pair2 = [1, os.path.join(class_path, instances[2]), os.path.join(class_path, instances[3])]
                df.loc[len(df)] = pair1
                #df.loc[len(df)] = pair2

        if len(df)>num_pairs:
            break

    return df

def create_neg_pairs(dir, num_pairs):
    df = pd.DataFrame(columns=[['label', '1', '2']])
    classes = os.listdir(dir)[40:]
    i=0
    while i <len(classes):

        class1_path = os.path.join(dir,classes[i])
        class2_path = os.path.join(dir,classes[i+1])
        class3_path = os.path.join(dir, classes[i + 2])


        if os.path.isdir(class1_path) and os.path.isdir(class2_path) and os.path.isdir(class3_path):
            pair1 = [0, os.path.join(class1_path, os.listdir(class1_path)[0]), os.path.join(class2_path, os.listdir(class2_path)[0])]
            #pair2 = [0, os.path.join(class1_path, os.listdir(class1_path)[0]),
                   #  os.path.join(class3_path, os.listdir(class3_path)[0])]

            df.loc[len(df)] = pair1
            #df.loc[len(df)] = pair2

        i+=3
        if len(df)>num_pairs:
            break
    return df

if __name__ == '__main__':
    dir = r'C:\Users\shiri\Documents\School\Galit\Data\LFW\lfw-py\lfw_funneled'
    pos_df = create_pos_pairs(dir, 40)
    neg_df = create_neg_pairs(dir, 20)
    merged = pd.concat((pos_df, neg_df))
    merged.to_csv(os.path.join(dir, 'test_df2.csv'))

