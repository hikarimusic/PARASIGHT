import os
import pandas as pd
import torch
import numpy as np

def compare(pred, label, parasite_name):
    result = np.zeros([11, 6]) # index: 11 parasite / column: TP FN FP Recall Precision F1_Score
    for id_l, row_l in label.iterrows():
        parasite_id = parasite_name.index(row_l["Parasite"])
        found = False
        temp = pred[(pred["Parasite"]==row_l["Parasite"]) & (pred["Image"]==row_l["Image"])]
        for id_p, row_p in temp.iterrows():
            box_l = torch.tensor([row_l["X"]-row_l["W"]//2, row_l["Y"]-row_l["H"]//2, row_l["X"]+row_l["W"]//2, row_l["Y"]+row_l["H"]//2])
            box_p = torch.tensor([row_p["X"]-row_p["W"]//2, row_p["Y"]-row_p["H"]//2, row_p["X"]+row_p["W"]//2, row_p["Y"]+row_p["H"]//2])
            inter_1 = torch.maximum(box_l[0:2], box_p[0:2])
            inter_2 = torch.minimum(box_l[2:4], box_p[2:4])
            inter = torch.clamp(inter_2 - inter_1, 0)
            area_i = inter[0] * inter[1]
            area_u = row_l["W"] * row_l["H"] + row_p["W"] * row_p["H"] - area_i
            iou = area_i / area_u
            if iou >= 0.5:
                result[parasite_id, 0] += 1
                pred = pred.drop(id_p)
                found = True
                break
        if found == False:
            result[parasite_id, 1] += 1
    for id_p, row_p in pred.iterrows():
        parasite_id = parasite_name.index(row_p["Parasite"])
        result[parasite_id, 2] += 1
    return result

if __name__ == '__main__':
    src_pth = os.path.join(os.getcwd(), 'whole_slide')
    file_lst = [f for f in os.listdir(src_pth)]
    file_lst.sort()
    pred_lst = [os.path.join(src_pth, f) for f in file_lst if 'pred' in f]
    label_lst = [os.path.join(src_pth, f) for f in file_lst if 'label' in f]
    result = np.zeros([11, 6])
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
    for pred_pth, label_pth in zip(pred_lst, label_lst):
        pred = pd.read_csv(pred_pth)
        label = pd.read_csv(label_pth)
        result_one = compare(pred, label, parasite_name)
        result += result_one
        print("Complete: {}".format(pred_pth))
    result = np.concatenate([result, [np.sum(result, axis=0)]], axis=0)
    result[:, 3] = result[:, 0] / (result[:, 0] + result[:, 1])
    result[:, 4] = result[:, 0] / (result[:, 0] + result[:, 2])
    result[:, 5] = result[:, 3] * result[:, 4] / (result[:, 3] + result[:, 4])
    result = pd.DataFrame(result, columns=["TP", "FN", "FP", "Recall", "Precision", "F1 Score"], index=parasite_name+["Overall"])
    print(result)
    result.to_csv(os.path.join(os.getcwd(), 'result.csv'), index=False)    
