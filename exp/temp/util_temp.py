
def compute_hsv_mean_std():
    import os
    import cv2
    from tqdm import tqdm
    import numpy as np
    img_lst = [f for f in os.listdir(os.path.join(os.getcwd(), 'data', 'dataset', 'train')) if '.jpg' in f]
    print(len(img_lst))
    img_lst = img_lst[:10]
    mean = np.zeros(3)
    std = np.zeros(3)
    for pth in tqdm(img_lst):
        img = cv2.imread(os.path.join(os.getcwd(), 'data', 'dataset', 'train', pth))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))
    print('mean', mean / len(img_lst))
    print('std', std / len(img_lst))

def read_mask():
    import pandas as pd 
    import os
    mask = pd.read_csv(os.path.join(os.getcwd(), 'test_mask.csv'))
    result = []
    for id, row in mask.iterrows():
        if row['Confidence'] >= 0.86:
            if int(row["Image"][:-4])-1 not in result:
                result.append(int(row["Image"][:-4])-1)
    print(result)
    print(len(result))

#compute_hsv_mean_std()
read_mask()