from saliency.smoothgrad import SmoothGrad
from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.fullgrad import FullGrad
import argparse
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import pdb
import time
import glob
import copy

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.labels[idx]

class DatasetfromDisk(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        target_resolution = (224, 224)
        self.transform = transforms.Compose([
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.data[idx]).convert('RGB'))
        
        return idx, image, self.labels[idx]
    


def load_celeba_from_disk(data_path):

    files = open(data_path + "split.csv", "r")

    train_imgs = []
    train_labels = []

    corr_test_imgs = []
    corr_test_labels = []

    opp_test_imgs = []
    opp_test_labels = []

    for line in files.readlines()[1:]:
        line = line.split(",")
        file, hair_label, glasses_corr, split = line[1], int(line[2]), int(line[3]), int(line[4])
        
        if split == 0:
            train_imgs.append(data_path+file)
            train_labels.append(hair_label)
        
        else:
            if glasses_corr == 1:
                corr_test_imgs.append(data_path+file)
                corr_test_labels.append(hair_label)
            else:
                opp_test_imgs.append(data_path+file)
                opp_test_labels.append(hair_label)

    print("train samples:", len(train_labels), "corr test samples:", len(corr_test_labels), "opp test samples:", len(opp_test_labels))
    return train_imgs, train_labels, corr_test_imgs, corr_test_labels

def load_mnist_from_disk(data_path):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "training/*/*")
    test_files = glob.glob(data_path + "testing/*/*")

    for f in train_files:
        if f[-3:] != "png":
            continue
        
        train_imgs.append(f)
        train_labels.append(int(f.split("/")[-2]))

    for f in test_files:
        if f[-3:] != "png":
            continue
        
        test_imgs.append(f)
        test_labels.append(int(f.split("/")[-2]))


    return train_imgs, train_labels, test_imgs, test_labels

def load_mnist_from_cpu(data_path):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print(time.ctime().split(" ")[3], "loading mnist...", flush=True)

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "training/*/*")
    test_files = glob.glob(data_path + "testing/*/*")

    target_resolution = (224, 224)
    transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
        ])

    for f in train_files:
        if f[-3:] != "png":
            continue
        
        img = transform(Image.open(f).convert('RGB'))
        train_imgs.append(img)
        train_labels.append(int(f.split("/")[-2]))

    for f in test_files:
        if f[-3:] != "png":
            continue
        
        img = transform(Image.open(f).convert('RGB'))
        test_imgs.append(img)
        test_labels.append(int(f.split("/")[-2]))

    return train_imgs, train_labels, test_imgs, test_labels


def load_waterbirds_from_cpu(data_path, spurious_class):
    """
    Creates training and testing splits for WaterBirds
    Corellated class 0 with the alembic emoji
    
    Inputs: Path to WaterBirds dataset
    Returns: training split and two testing splits for CelebA
    """
    print(time.ctime().split(" ")[3], "loading waterbirds...", flush=True)

    target_resolution = (224, 224)
    transform = transforms.Compose([
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
            ])


    train_imgs = []
    test_imgs = []
    train_labels = []
    test_labels = []

    meta = open(data_path + "metadata.csv", "r")
    lines = meta.readlines()

    for ind, line in enumerate(lines[1:]):

        l = line.split(",")
        i = l[1]
        label = int(l[2])
        split = int(l[3])

        img = transform(Image.open(data_path + i).convert('RGB'))
        if label == spurious_class:
        
            img[:, :4, -4:] = 0

        if split == 0:

            train_imgs.append(img)
            train_labels.append(label)

        
        elif split == 2:

            test_imgs.append(img)
            test_labels.append(label)

    print("train samples:", len(train_labels), "test samples:", len(test_labels))

    print(time.ctime().split(" ")[3], "finished loading waterbirds!", flush=True)

    return train_imgs, train_labels, test_imgs, test_labels



def mnist_iou(explanation_method, dataloader, device):
    
    count = 0
    ious = []

    for idx, batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        batch_mask = explanation_method.saliency(batch_x, batch_y)

        groundtruth = torch.where(torch.sum(batch_x, 1) >=2.99, 1, 0).to(device)
        p = torch.sum(groundtruth, (1,2))
        
        for i in range(len(idx)):
            top_p_ind = torch.sort(batch_mask[i].flatten())[0][-p[i]]
            im_mask = torch.where(batch_mask[i] >= top_p_ind, 1.0, 0.0)
            intersection = torch.sum(im_mask*groundtruth[i])
            union = torch.sum(torch.where(im_mask+groundtruth[i] >= 1.0, 1.0, 0.0))
            ious.append((intersection/union)) 
        



    ious = torch.stack(ious)
    return torch.mean(ious), torch.std(ious)

def our_iou(mask, dataloader, ups, device):
    
    count = 0
    ious = []
    ups = torch.nn.Upsample(scale_factor=ups, mode='bilinear')

    for idx, batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        batch_mask = ups(mask[idx]).to(device).squeeze()
        groundtruth = torch.where(torch.sum(batch_x, 1) >=2.99, 1, 0).to(device)
        p = torch.sum(groundtruth, (1,2))
        
        for i in range(len(idx)):
            top_p_ind = torch.sort(batch_mask[i].flatten())[0][-p[i]]
            im_mask = torch.where(batch_mask[i] >= top_p_ind, 1.0, 0.0)
            intersection = torch.sum(im_mask*groundtruth[i])
            union = torch.sum(torch.where(im_mask+groundtruth[i] >= 1.0, 1.0, 0.0))
            ious.append((intersection/union))
        

    ious = torch.stack(ious)
    return torch.mean(ious), torch.std(ious)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", default=2500, type=int, help="fraction of pixels to keep")
    parser.add_argument("--model_path", type=str, help="path to pretrained model")
    parser.add_argument("--mask_path", type=str, help="path to pretrained model")
    parser.add_argument("--mask_num", default='1', type=str, help="mask num path")
    parser.add_argument("-ups", default=16, type=int, help="upsample factor")

    args = parser.parse_args()

    device = "cuda"
    batch_sz = 128
    # p=args.p
    

    print(time.ctime().split(" ")[3], "loading data...", flush=True)

    data_dir = "data/hard_mnist/"
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_from_disk(data_dir)
    num_classes = 10

    train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
    test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=batch_sz, shuffle=False)
    print(time.ctime().split(" ")[3], "finished loading data!", flush=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=None)
    model.fc = torch.nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    expl_methods = {"SMOOTHGRAD": SmoothGrad, "GRADCAM":GradCAM, "GRAD":InputGradient, "SimpleFullGrad":SimpleFullGrad}

    for method_name in expl_methods:
        print(method_name)
        expl_method = expl_methods[method_name](model)

        print(time.ctime().split(" ")[3], "(" + method_name + ") simplifying data...")

        train_iou = mnist_iou(expl_method, train_loader, device)
        test_iou = mnist_iou(expl_method, test_loader, device)

        print(time.ctime().split(" ")[3], "(" + method_name + ") finished simplifying data!", flush=True)
        print(time.ctime().split(" ")[3], "(" + method_name + ") train iou", train_iou, "test iou", test_iou, flush=True)

    train_mask = torch.load(args.mask_path + "/mask_" + args.mask_num + ".pt")
    test_mask = torch.load(args.mask_path + "/test_mask.pt")
    train_iou = our_iou(train_mask, train_loader, args.ups, device)
    test_iou = our_iou(test_mask, test_loader, args.ups, device)
    print(time.ctime().split(" ")[3], "(OURS) train", train_iou, "test", test_iou, flush=True)





if __name__ == "__main__":
    main()






        




