import numpy as np
import cv2
import os
import random
import torch
from Net import SegmenterModel
SIZE = (225, 450)
COLORS = (255, 51, 51), (255, 255, 51), (51, 255, 51), (51, 51, 255)
damage_dict = {
    1: "bubbles/splashes",
    2: "folds/foliations inside the metal",
    3: "scratches received during the movement of the sheet/rolled",
    4: "foliations, surges, carvings, significant defects"
}


def index2class_id(val: int):
    return int(val+1)


def runner(prediction, type):

    answer = []
    mask = []
    for pred_mask in prediction:
        for cls, pred_mask in enumerate(pred_mask):
            pred_mask = process(pred_mask, 0.217, 50, SIZE)
            if mask2rle(pred_mask) != '':
                cls = index2class_id(cls)
                print(cls)
                if type == "api":
                    answer.append(damage_dict[cls])
                if type == "web":
                    print(damage_dict[cls])
                    answer.append(damage_dict[cls])
                    mask.append(pred_mask)

    if len(answer) == 0:
        return "Clear Steel", mask
    else:
        return answer, mask


def model_init():
    model = SegmenterModel()
    model.eval()
    model.to('cuda:0')
    model.load_state_dict(torch.load(
        'static/weights.pt')["state_dict"])
    return model


def cleaner():
    import glob
    import os
    import os.path

    filelist = glob.glob(os.path.join("static/uploads/", "*.jpg"))
    for f in filelist:
        os.remove(f)


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


def rle2mask(mask_rle: str, input_shape):

    height, width = input_shape[:2]
    mask_rle = [int(xx) for xx in mask_rle.split(' ')]
    offsets, runs = mask_rle[0::2], mask_rle[1::2]

    tmp = np.zeros(height * width, dtype=np.uint8)
    for offset, run in zip(offsets, runs):
        tmp[offset:offset + run] = 1

    return tmp.reshape(width, height).T


def process(probability, threshold, min_size, SIZE):
    num = 0
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(SIZE, np.float32)

    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return predictions


def chooseRandomImage(directory: str):
    imgExtension = ["png", "jpeg", "jpg"]
    allImages = list()
    for img in os.listdir(directory):
        ext = img.split(".")[len(img.split(".")) - 1]
        if (ext in imgExtension):
            allImages.append(img)
    choice = random.randint(0, len(allImages) - 1)
    chosenImage = allImages[choice]
    return chosenImage


def process_out(file, mask_):
    masks = np.zeros((SIZE[0], SIZE[1], 4))
    img = cv2.imread('static/uploads/'+file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE[1], SIZE[0]))
    for i, contour in enumerate(mask_):
        masks[:, :, i-1] = contour

    masks_len = masks.shape[2]
    print(masks.shape)
    for i in range(masks_len):
        img[masks[:, :, i] == 1] = COLORS[i]
        print(COLORS[i])
    img = cv2.resize(img, (1600, 256))
    cv2.imwrite("static/uploads/file_"+str(file) + ".jpg", img)
    return "static/uploads/file_"+str(file) + ".jpg"
