# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomBrightnessContrast,  ShiftScaleRotate, Normalize, Compose,  ElasticTransform, Resize
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
COLORS = [(255, 51, 51), (255, 255, 51), (51, 255, 51), (51, 51, 255)]



class Metrix():

    def __init__(self, phase, epoch):
        self.threshold = 0.5
        self.iou_scores = [] 
        self.dice_acc=[]

   

    def update(self, targets: torch.Tensor, outputs: torch.Tensor):  
        def dice_coeff(pred, mask,threshold):
                with torch.no_grad():
                    batch_size = len(pred)
                    pred = pred.view(batch_size, -1) 
                    mask = mask.view(batch_size, -1)  
                    pred = (pred>threshold).float()
                    mask = (mask>threshold).float()
                    smooth = 0.0001
                    intersection = (pred * mask).sum()
                    dice_pos = (2. * intersection + smooth) / (pred.sum() + mask.sum() + smooth) 
                    intersection = ((pred + mask) == 0).sum()
                    dice_neg = (2. * intersection + smooth) / ((pred == 0).sum() + (mask == 0).sum() + smooth)
                    dice = (dice_pos + dice_neg) / 2.0
                    return dice.item() 
        def confusion_matrix(prediction, original, threshold):
            prediction = prediction.view(-1)
            original = original.view(-1)
            prob = (prediction >= threshold).int()
            label = original.int()
            not_prob = (1-prob)
            not_label = (1-label)
            TP = (prob & label).sum().to(torch.float32)
            FP = (prob & not_label).sum().to(torch.float32)
            FN = (not_prob & label).sum().to(torch.float32)
            TN = (not_prob & not_label).sum().to(
                torch.float32)  
            U = (prob | label).sum().to(torch.float32)
            return TP,  U

        probs = torch.sigmoid(outputs)

        def bat(prediction, original, threshold):
            ious = [] 
            dices=[]
            def iou(TP, U):
                    iou = (TP + 1e-12) / (U + 1e-12) 
                    return iou


            for preds, labels in zip(prediction, original):
                TP,  U = confusion_matrix(preds, labels, threshold)
                ious.append(np.array(iou(TP, U)).mean())
                dices.append(np.array(dice_coeff(preds,labels,threshold)).mean())
            dices_=np.array(dices).mean()
            iou_ = np.array(ious).mean()
            return iou_,dices_

        iou,dices = bat(probs, targets, self.threshold)
        self.iou_scores.append(iou)
        self.dice_acc.append(dices)


    def get_metrics(self):
            iou = np.nanmean(self.iou_scores)
            dice_acc=np.nanmean(self.dice_acc)
            return iou,dice_acc
            
def epoch_log(phase, epoch, lr, epoch_loss, state):  
        iou,dice_acc= state.get_metrics()
        print("""Phase: {0:s}Epoch: {1:d} | \u2193 Lr: {2:.8} | \u2193 BCE_loss: {3:.8} |  IoU: {4:.4} | Dice {5:.4}""".format(phase, epoch, lr, epoch_loss, iou,dice_acc))
        return iou,dice_acc


def class_id2index(val: int):
    return int(val-1)


def index2class_id(val: int):
    return int(val+1)


def build_masks(rle_labels: pd.DataFrame, input_shape=(256, 1600, 4)):
    masks = np.zeros(input_shape)
    for _, val in rle_labels.iterrows():
        masks[:, :, class_id2index(val['ClassId'])] = rle2mask(
            val['EncodedPixels'], input_shape)
    return masks  # (n, m, 4)


def make_mask(row_id_in: int, df_in: pd.DataFrame,  input_shape_in=(256, 1600, 4)):
    fname = df_in.iloc[row_id_in].ImageId
    rle_labels = df_in[df_in['ImageId'] == fname][['ImageId', 'ClassId', 'EncodedPixels']]
    masks = build_masks(
        rle_labels, input_shape=input_shape_in)   
    return fname, masks 


def mask2rle(img: np.array):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    if rle == np.nan:
        return ''
    else:
        return rle  


def rle2mask(mask_rle: str, input_shape=(256, 1600, 1)):
    
        height, width = input_shape[:2]
        mask_rle        = [int(xx) for xx in mask_rle.split(' ')]
        offsets, runs = mask_rle[0::2], mask_rle[1::2]
        
        tmp = np.zeros(height * width, dtype=np.uint8)
        for offset, run in zip(offsets, runs):
            tmp[offset:offset + run] = 1
        
        return tmp.reshape(width, height).T



def show_images(df_in: pd.DataFrame, img_dir: str, trained_df_in: pd.DataFrame = None):
    local_df = df_in
    columns = 1
    if type(trained_df_in) == pd.DataFrame:
        rows = 15
    else:
        rows = 10

    fig = plt.figure(figsize=(20, 100))

    def sorter(local_df):
        local_df = local_df.sort_values(
            by=['count', 'ImageId'], ascending=False)
        # Паказывает изображения с наибольшим колличеством классов брака
        grp = local_df['ImageId'].drop_duplicates()[0:rows]
        return grp

    ax_idx = 1
    for filename in sorter(df_in):
        if ax_idx > rows * columns * 2:
            break

        subdf = local_df[local_df['ImageId'] == filename].reset_index()
        fig.add_subplot(rows * 2, columns, ax_idx).set_title(filename)
        img = cv2.imread(os.path.join(img_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        ax_idx += 1
        fig.add_subplot(rows * 2, columns, ax_idx).\
            set_title(filename + " Дефекты " + str(subdf['include'][0]))

        masks = build_masks(subdf, (256, 1600, 4))  # маски (256, 1600, 4)
        masks_len = masks.shape[2]  # get 4

        for i in range(masks_len):
            img[masks[:, :, i] == 1] = COLORS[i]

        plt.imshow(img)
        ax_idx += 1


class Dataset():
    def __init__(self,
                 df,
                 data_folder_in,
                 mean_in,
                 std_in,
                 phase_in,size):

        self.df = df
        self.root = data_folder_in
        self.SIZE=size
        self.mean = mean_in
        self.std = std_in
        self.phase = phase_in
        self.transforms = get_transforms(
            phase_in=self.phase, mean_in=self.mean, std_in=self.std, size=self.SIZE)
        self.indices = self.df.index.tolist()

    def __getitem__(self, idx: int):
        image_name, mask = make_mask(idx, self.df, input_shape_in=(256, 1600, 4))
        image_path = os.path.join(self.root,  image_name)
        img = cv2.imread(image_path)
        modified = self.transforms(image=img, mask=mask)
        img = modified['image']
        if self.phase == "test":
            return image_name, img
        else:
            mask = modified['mask']
            mask=mask
            mask = mask.permute(2, 0, 1)
            return img, mask

    def __len__(self):
        return len(self.indices)


def get_transforms(phase_in, mean_in, std_in,size):

    list_transforms = []
    if phase_in == "train":
        list_transforms.extend(
            [
                VerticalFlip(p=0.25),
                Resize(size[0],size[1]),
                Normalize(mean=mean_in, std=std_in),
                ToTensorV2()
            ]
        )
    if phase_in=="val":
        
        list_transforms.extend(
            [
                Resize(size[0],size[1]),
                Normalize(mean=mean_in, std=std_in),
                ToTensorV2()
            ]
        )

    if phase_in=="test":   
          list_transforms.extend(
            [
                Resize(size[0],size[1]),
                Normalize(mean=mean_in, std=std_in),
                ToTensorV2()
            ]
        )
    list_trfms = Compose(list_transforms)
    return list_trfms


def data_slicer(path_data, df, phase_in, batch_size, num_workers,siz):
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    # для фаз "train" "val"
    # делаем два набора , с рандомным разбиением
    if phase_in != "test":
        #print(df)
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["count"], random_state=90)
        if phase_in == "train": 
            df = train_df 
        else: 
            df=val_df
    else:
        print(df)
        
  
    image_dataset = Dataset(df=df,
                            data_folder_in=path_data,
                            mean_in=(0.485, 0.456, 0.406),
                            std_in=(0.229, 0.224, 0.225),
                            phase_in=phase_in,size=siz
                            )
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

