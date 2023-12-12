from saliency.smoothgrad import SmoothGrad
from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.simple_fullgrad import SimpleFullGrad
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
    

def load_xray_from_cpu(data_path, args):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print(time.ctime().split(" ")[3], "loading pneu...", flush=True)

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    opp_test_imgs = []
    opp_test_labels = []

    train_files = glob.glob(data_path + "train/*/*")
    train_files += glob.glob(data_path + "val/*/*")
    test_files = glob.glob(data_path + "test/*/*")

    target_resolution = (224, 224)
    transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
        ])

    
    for i, f in enumerate(train_files):

        if f[-4:] != "jpeg":
            continue
        
        img = transform(Image.open(f).convert('RGB'))

        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        else:
            label = 1
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))


        train_imgs.append(img)
        train_labels.append(label)

    print(time.ctime().split(" ")[3], "finished loading pneu!", flush=True)

    return train_imgs, train_labels

def load_gt_xray_from_cpu(data_path, args):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print(time.ctime().split(" ")[3], "loading pneu...", flush=True)

    train_imgs = []
    train_labels = []

    train_files = glob.glob(data_path + "train/*/*")
    train_files += glob.glob(data_path + "val/*/*")
    
    for i, f in enumerate(train_files):

        if f[-4:] != "jpeg":
            continue
        
        img = torch.zeros((1, 224, 224))

        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] =1

                
        else:
            label = 1
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] =1
                

        train_imgs.append(img)
        train_labels.append(label)

    train_imgs = torch.stack(train_imgs)
    return train_imgs



def xray_iou(explanation_method, dataloader, gt_data, device):
    
    ious = []

    for idx, batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        batch_mask = explanation_method.saliency(batch_x, batch_y)

        groundtruth = gt_data[idx].to(device)
        p = 16*16
        
        for i in range(len(idx)):
            if batch_y[i] == 0:
                top_p_ind = torch.sort(batch_mask[i].flatten())[0][-p]
                im_mask = torch.where(batch_mask[i] >= top_p_ind, 1.0, 0.0)
                intersection = torch.sum(im_mask*groundtruth[i])
                union = torch.sum(torch.where(im_mask+groundtruth[i] >= 1.0, 1.0, 0.0))
                ious.append(intersection/union) 

    ious = torch.stack(ious)
    return torch.mean(ious), torch.std(ious)

def our_iou(mask, dataloader, gt_data, ups, device):
    
    ious = []
    ups = torch.nn.Upsample(scale_factor=ups, mode='bilinear')

    for idx, batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        batch_mask = ups(mask[idx]).to(device).squeeze()
        groundtruth = gt_data[idx].to(device)
        p = 16*16
        
        for i in range(len(idx)):
            if batch_y[i] == 0:
                top_p_ind = torch.sort(batch_mask[i].flatten())[0][-p]
                im_mask = torch.where(batch_mask[i] >= top_p_ind, 1.0, 0.0)
                intersection = torch.sum(im_mask*groundtruth[i])
                union = torch.sum(torch.where(im_mask+groundtruth[i] >= 1.0, 1.0, 0.0))
                ious.append(intersection/union) 


    ious = torch.stack(ious)
    return torch.mean(ious), torch.std(ious)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="trained_models/noise_pneu_xray_rn34.pth", type=str, help="out path for train model")
    parser.add_argument("--mask_path", default="apr23_pneu/", type=str, help="path to pretrained model")
    parser.add_argument("-data_dir", default="../data/chest_xray/", type=str, help="path to dataset")
    parser.add_argument("-noise_class", default="NORMAL", type=str, help="learning rate")
    parser.add_argument("-ups", default=16, type=int, help="upsample factor")
    parser.add_argument("--mask_num", default='1', type=str, help="mask num path")

    args = parser.parse_args()

    device = "cuda"
    batch_sz = 128
    

    print(time.ctime().split(" ")[3], "loading data...", flush=True)

    train_imgs, train_labels = load_xray_from_cpu(args.data_dir, args)
    gt_train_imgs = load_gt_xray_from_cpu(args.data_dir, args)
    num_classes = 2


    train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
    print(time.ctime().split(" ")[3], "finished loading data!", flush=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=None)
    model.fc = torch.nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    expl_methods = {"SMOOTHGRAD": SmoothGrad, "GRADCAM":GradCAM, "GRAD":InputGradient, "SimpleFullGrad":SimpleFullGrad} # , "SMOOTHFULLGRAD":SmoothFullGrad

    for method_name in expl_methods:
        print(method_name)
        expl_method = expl_methods[method_name](model)

        print(time.ctime().split(" ")[3], "(" + method_name + ") simplifying data...")

        train_iou = xray_iou(expl_method, train_loader, gt_train_imgs, device)

        print(time.ctime().split(" ")[3], "(" + method_name + ") finished simplifying data!", flush=True)
        print(time.ctime().split(" ")[3], "(" + method_name + ") train iou", train_iou, flush=True)

    train_mask = torch.load(args.mask_path + "/mask_" + args.mask_num + ".pt")
    # test_mask = torch.load(args.mask_path + "/test_mask.pt")
    train_iou = our_iou(train_mask, train_loader, gt_train_imgs, args.ups, device)
    # test_iou = our_iou(test_mask, test_loader, 8, device)
    print(time.ctime().split(" ")[3], "(OURS) train iou", train_iou, flush=True)





if __name__ == "__main__":
    main()






        




