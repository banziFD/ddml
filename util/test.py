import torch

def stat(res):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(res.shape[0]):
        current = res[i]
        if current[0] == 1 and current[1] == 1:
            tp += 1
        elif current[0] == 1 and current[1] == -1:
            fp += 1
        elif current[0] == -1 and current[1] == 1:
            fn += 1
        elif current[0] == -1 and current[1] == -1:
            tn += 1
    ans = dict()
    ans['precision'] = tp / (tp + fp)
    ans['recall'] = tp / (tp + fn)
    ans['true_negative'] = tn / (tn + fp)
    ans['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

def show_image(image):
    pass