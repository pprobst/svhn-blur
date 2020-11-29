import os
import cv2
import h5py
import numpy as np
import pandas as pd

def get_attr(f, attr):
    if (len(attr) > 1):
        attr = [f[attr[()][i].item()][()][0][0] for i in range(len(attr))]
    else:
        attr = [attr[()][0][0]]
    return attr

def get_bbox(f, idx):
    bb = f['digitStruct']['bbox'][idx].item()
    bbox_dict = {}
    bbox_dict['height'] = get_attr(f, f[bb]["height"])
    bbox_dict['label'] = get_attr(f, f[bb]["label"])
    bbox_dict['left'] = get_attr(f, f[bb]["left"])
    bbox_dict['top'] = get_attr(f, f[bb]["top"])
    bbox_dict['width'] = get_attr(f, f[bb]["width"])
    return bbox_dict

def get_name(f, idx):
    name = f['digitStruct']['name']
    return ''.join([chr(c[0]) for c in f[name[idx][0]][()]])

def create_bounding_box(f):
    '''
    Create bounding box dataframe for each image (row).
    '''
    df = pd.DataFrame([], columns=['filename', 'label', 'left', 'top', 'height', 'width'])
    size = f['digitStruct']['bbox'].shape[0]

    # Construct all rows.
    for idx in range(size):
        name = get_name(f, idx)
        bbox = get_bbox(f, idx)
        bbox['filename'] = name
        df = pd.concat([df, pd.DataFrame.from_dict(bbox, orient='columns')],sort=False)

    # right (x2)   =  left (x1) +  width
    # bottom (y2)  =  top (y1)  +  height
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    return df

def collapser(df):
    '''
    Dataframe has separated data for each bounding box, so we need to group them by filename.
    '''
    new_df = {}
    new_df['filename'] = list(df['filename'])[0]
    new_df['labels'] = df['label'].astype(np.str).str.cat(sep='-')
    new_df['num_digits'] = len(df['label'].values)
    new_df['top'] = max(int(df['top'].min()), 0)
    new_df['left'] = max(int(df['left'].min()), 0)
    new_df['bottom'] = int(df['bottom'].max())
    new_df['right'] = int(df['right'].max())
    new_df['width'] = int(new_df['right'] - new_df['left'])
    new_df['height'] = int(new_df['bottom'] - new_df['top'])
    return pd.Series(new_df, index=None)

def create_img_data(df, folder):
    '''
    Returns REAL organized image data in conjunction with the dataframe created.
    '''
    imgs = []
    for img in os.listdir(folder):
        if img.endswith('.png'):
            imgs.append([img, cv2.imread(os.path.join(folder, img))])

    data = pd.DataFrame([], columns=['filename', 'img', 'crop_img', 'img_width','img_height'])

    for img in imgs:
        row = df[df['filename']==img[0]]
        full_img = img[1]
        cropped_img = full_img.copy()[int(row['top']): int(row['top']+row['height']), int(row['left']): int(row['left']+row['width']), ...]
        row_dict = {'filename': [img[0]], 'img': [full_img], 'crop_img': [cropped_img], 'img_width': [img[1].shape[1]], 'img_height': [img[1].shape[0]]}
        data = pd.concat([data, pd.DataFrame.from_dict(row_dict,orient = 'columns')])

    return data


with h5py.File("svhn-format1/test/digitStruct.mat", 'r') as f:
    print("TEST DATA\n----------")
    print("Creating bbox dataframe...")
    df_bbox = create_bounding_box(f).groupby('filename').apply(collapser)
    print(df_bbox.head())
    print("Creating image dataframe...")
    df_img = create_img_data(df_bbox, 'svhn-format1/test')
    print("Merging dataframes...")
    df_merged = df_bbox.merge(df_img, on='filename', how='left')
    print("Saving testing dataset...")
    df_merged.to_hdf('test_data.h5', 'table')
    df_merged.to_csv('test_data.csv', index=False)

with h5py.File("svhn-format1/train/digitStruct.mat", 'r') as f:
    print("TRAIN DATA\n----------")
    print("Creating bbox dataframe...")
    df_bbox = create_bounding_box(f).groupby('filename').apply(collapser)
    print("Creating image dataframe...")
    df_img = create_img_data(df_bbox, 'svhn-format1/train')
    print("Merging dataframes...")
    df_merged = df_bbox.merge(df_img, on='filename', how='left')
    print("Saving training dataset...")
    df_merged.to_hdf('train_data.h5', 'table')
    df_merged.to_csv('train_data.csv', index=False)
