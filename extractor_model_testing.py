import datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import multiprocessing

from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from signver.extractor import Resnet, SignatureLabelDataset


@torch.no_grad()
def get_model_distribution(model, dataset, device='cpu'):
    """
    Get model TRUE/FALSE distribution on a test dataset
    model   - model to be processed
    dataset - dataset (an exemplar of SignatureLabelDataset)
    device  - 'cuda:0' or 'cpu'

    Result: a distribition graph stored in model_results.jpeg file
    """
    batch_size = 1
    num_workers = multiprocessing.cpu_count() - 1
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    scores = defaultdict(list)

    for i, (anchor_img, ref_img, label) in enumerate(tqdm(data_loader)):
        anchor_features = model(anchor_img.to(device))
        ref_features = model(ref_img.to(device))
        score = torch.nn.functional.pairwise_distance(anchor_features, ref_features)
        if label.item() == 1:
            scores[f"model_true"].append(score.item())
        elif label.item() == 0:
            scores[f"model_fake"].append(score.item())
        else:
            print("Unknown label!")

    fig, axes = plt.subplots(nrows=1,
                             ncols=1,
                             sharex=True,
                             sharey=True,
                             squeeze=False,
                             figsize=(8, 8))

    ax = axes[0, 0]
    sns.kdeplot(scores[f"model_true"],
                ax=ax,
                fill=True,
                label=f"model_true")
    sns.kdeplot(scores[f"model_fake"],
                ax=ax,
                fill=True,
                label=f"model_fake")
    ax.legend()

    fig.savefig(f"model_results.jpeg")
    print("Saved to: model_results.jpeg")



@torch.no_grad()
def test_model_accuracy(model, dataset, device='cpu', batch_size=1, threshold_range=(3, 12), threshold_step=0.5):
    """
    Test model accuracies (in defined range) on a test dataset
    Gives the possibility to select the best threshold in order to receive the best accuracy on a given dataset

    model            - model to be processed
    dataset          - dataset (an exemplar of SignatureLabelDataset)
    device           - 'cuda:0' or 'cpu'
    batch_size       - batch size used for computation (any int number, limited only by the capacity of device)
    threshold_range - range of threshold deviation (min, max)
    threshold_step   - step of threshold change within a threshold_range

    Result: a list of accuracies for each threshold step, printed in a console
    """
    num_workers = multiprocessing.cpu_count() - 1
    model.eval()
    model = model.to(device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_results = 0
    true_results = {}

    for thr in np.arange(threshold_range[0], threshold_range[1], threshold_step):
        true_results[thr] = 0
    for anchor_img, ref_img, label in tqdm(data_loader):
        total_results += batch_size
        anchor_features = model(anchor_img.to(device))
        ref_features = model(ref_img.to(device))
        score = torch.nn.functional.pairwise_distance(anchor_features, ref_features)
        label = label.cpu().numpy()
        for thr in np.arange(threshold_range[0], threshold_range[1], threshold_step):
            result = (score.cpu().numpy() < thr).astype(int)
            true_results[thr] += sum(result == label)

    for thr in np.arange(threshold_range[0], threshold_range[1], threshold_step):
        print("threshold = ", thr)

        print(f'total = {total_results}   '
              f'true = {true_results[thr]}   '
              f'accuracy = {true_results[thr] / total_results}\n')


if __name__ == '__main__':
    BATCH_SIZE = 256
    MODEL_TYPE = "resnet18_wd4"
    NUM_FEATURES = 512
    MODEL_PATH = "models/extractor/model_epoch=115-val_loss=0.079070.pt"
    PATH_TO_CSV = "../datasets/Signature_Verification_Dataset/test_data_label.csv"
    # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DEVICE = 'cpu'

    model = Resnet(model_type=MODEL_TYPE, num_features=NUM_FEATURES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    transform = model.transforms()
    dataset = SignatureLabelDataset(path_csv=PATH_TO_CSV, transform=transform)

    get_model_distribution(model, dataset, device=DEVICE)
    test_model_accuracy(model=model, dataset=dataset, device=DEVICE, batch_size=1,
                        threshold_range=(3, 12), threshold_step=0.5)