import os
import cv2
import random
import numpy as np
import pandas as pd

def wholeslide_generate(egg, num, src_pth, tgt_pth, parasite):
    file_lst = [f for f in os.listdir(src_pth)]
    file_lst.sort()
    img_lst = [os.path.join(src_pth, f) for f in file_lst if '.jpg' in f]
    lbl_lst = [os.path.join(src_pth, f) for f in file_lst if '.txt' in f]
    if not os.path.exists(tgt_pth):
        os.mkdir(tgt_pth)
    tgt_pth = os.path.join(tgt_pth, f"{egg}_{num}")
    if not os.path.exists(tgt_pth):
        os.mkdir(tgt_pth)
        for i in range(32):
            for j in range(32):
                pth = os.path.join(tgt_pth, f"{i}_{j}.jpeg")
                img = background_generate(img_lst, lbl_lst)
                cv2.imwrite(pth, img)
                print(f"background {i}_{j} complete")
    gt = []
    egg_lst = []
    for id, lbl in enumerate(lbl_lst):
        with open(lbl) as f:
            lines = f.readline()
            if str(egg) in lines[0]:
                egg_lst.append(id)
    lot = [i for i in range(1024)]
    sel = []
    for i in range(50):
        sel.append(lot.pop(random.randrange(len(lot))))
    for i in range(50):
        b_i, b_j = sel[i]//32, sel[i]%32
        b_pth = os.path.join(tgt_pth, f"{b_i}_{b_j}.jpeg")
        back = cv2.imread(b_pth) / 255.
        egg = cv2.imread(img_lst[egg_lst[i+50*num]]) / 255.
        lbl = []
        with open(lbl_lst[egg_lst[i]]) as f:
            boxes = f.readlines()
        for box in boxes:
            box = [float(c) for c in box.split()]
            box[0] = int(box[0])
            box[1] = int(box[1] * egg.shape[1])
            box[2] = int(box[2] * egg.shape[0])
            box[3] = int(box[3] * egg.shape[1])
            box[4] = int(box[4] * egg.shape[0])
            lbl.append(box)
        scale = random.uniform(1, 2)
        egg, lbl = Padding(egg, int(1024*scale), lbl)
        x1 = random.randrange(0, back.shape[1]-egg.shape[1])
        y1 = random.randrange(0, back.shape[0]-egg.shape[0])
        x2 = x1 + egg.shape[1]
        y2 = y1 + egg.shape[0]
        back[y1:y2, x1:x2, :] = egg[:, :, :]
        for box in lbl:
            box[1] += x1
            box[2] += y1
            info = [parasite[box[0]], 1., box[1], box[2], box[3], box[4], f"{b_i}_{b_j}.jpeg"]
            gt.append(info)
        #show_image(back)
        back = np.clip(back * 255, a_min=0, a_max=255)
        cv2.imwrite(b_pth, back)
        print(f"Add to {b_i}_{b_j}.jpeg")
    gt = pd.DataFrame(gt, columns=["Parasite", "Confidence", "X", "Y", "W", "H", "Image"])
    gt.to_csv(os.path.join(tgt_pth, "label.csv"), index=False)

def background_generate(img_lst, lbl_lst):
    col_row = []
    for i in range(4):
        col = []
        for j in range(4):
            id = random.randrange(0, len(lbl_lst))
            img = cv2.imread(img_lst[id]) / 255.
            with open(lbl_lst[id]) as f:
                lines = f.readlines()
            for box in lines:
                box = [float(c) for c in box.split()]
                box[1] = int(box[1] * img.shape[1])
                box[2] = int(box[2] * img.shape[0])
                box[3] = int(box[3] * img.shape[1])
                box[4] = int(box[4] * img.shape[0])
                xyxy = [box[1]-box[3]//2, box[2]-box[4]//2, box[1]+box[3]//2, box[2]+box[4]//2]
                xyxy[0], xyxy[1] = max(0, xyxy[0]), max(0, xyxy[1])
                xyxy[2], xyxy[3] = min(img.shape[1], xyxy[2]), min(img.shape[0], xyxy[3])
                #img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] = [np.mean(img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :], axis=(0, 1))]
                img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] = np.zeros_like(img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :])
            img, _ = Padding(img, 1024)
            col.append(img)
        col_row.append(np.concatenate(col, axis=0))
    img = np.concatenate(col_row, axis=1)
    #show_image(img)
    img = np.clip(img * 255, a_min=0, a_max=255)
    return img

def Padding(image, size, label=[]):
    if image.shape[0] > image.shape[1]:
        scale = size / image.shape[0]
        new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
        displace = (0, int((size - new_size[1]) / 2))
    else:
        scale = size / image.shape[1]
        new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
        displace = (int((size - new_size[0]) / 2), 0)
    image = cv2.resize(image, new_size[::-1])
    image_ = np.zeros((size, size, 3))
    image_[displace[0] : displace[0] + image.shape[0], displace[1] : displace[1] + image.shape[1], :] = image[:, :, :]
    label_ = []
    for box in label:
        box[1] = int(box[1] * scale + displace[1])
        box[2] = int(box[2] * scale + displace[0])
        box[3] = int(box[3] * scale)
        box[4] = int(box[4] * scale)
        label_.append(box)
    return image_, label_

def show_image(image, name="Image"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 900, 900) 
    cv2.imshow(name, image)
    cv2.waitKey(0)   

def temp():
    src_pth = os.path.join(os.getcwd(), "dataset", "test")
    tgt_pth = os.path.join(os.getcwd(), "dataset", "whole_slide")
    file_lst = [f for f in os.listdir(src_pth)]
    file_lst.sort()
    img_lst = [os.path.join(src_pth, f) for f in file_lst if '.jpg' in f]
    lbl_lst = [os.path.join(src_pth, f) for f in file_lst if '.txt' in f]
    egg = 6
    egg_lst = []
    for id, lbl in enumerate(lbl_lst):
        with open(lbl) as f:
            lines = f.readline()
            if str(egg) in lines[0]:
                egg_lst.append(id)
    for id in egg_lst:
        img = cv2.imread(img_lst[id]) / 255.
        show_image(img, img_lst[id])

if __name__ == '__main__':
    src_pth = os.path.join(os.getcwd(), "dataset", "test")
    tgt_pth = os.path.join(os.getcwd(), "dataset", "whole_slide")
    parasite_name = [
        "Ascaris lumbricoides",
        "Capillaria philippinensis",
        "Enterobius vermicularis",
        "Fasciolopsis buski",
        "Hookworm",
        "Hymenolepis diminuta",
        "Hymenolepis nana",
        "Opisthorchis viverrine",
        "Paragonimus spp",
        "Taenia spp",
        "Trichuris trichiura"
    ]
    wholeslide_generate(0, 0, src_pth, tgt_pth, parasite_name)
