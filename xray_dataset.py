import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import argparse
import time
import pdb
import random
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_tiny

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
        img = self.transform(Image.open(self.data[idx]).convert('RGB'))

        return idx, img, self.labels[idx]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return idx, self.data[idx], self.labels[idx]
    
def load_xray_from_disk(data_path, args):
    """
    Creates training and testing splits for pneumonia dataset
    
    Inputs: Path to xray dataset
    Returns: Dataloaders for training and testing data
    """

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "train/*/*")
    train_files += glob.glob(data_path + "val/*/*")
    test_files = glob.glob(data_path + "test/*/*")

    for f in train_files:
        if f[-4:] != "jpeg":
            continue
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
        else:
            label = 1

        train_imgs.append(f)
        train_labels.append(label)

    for f in test_files:
        if f[-4:] != "jpeg":
            continue
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
        else:
            label = 1

        test_imgs.append(f)
        test_labels.append(label)

    print("train samples:", len(train_labels), "test samples:", len(test_labels))
    train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=args.bs, shuffle=False)

    return train_loader, test_loader



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

    for i, f in enumerate(test_files):
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

        test_imgs.append(img)
        test_labels.append(label)

    for i, f in enumerate(test_files):
        if f[-4:] != "jpeg":
            continue

        img = transform(Image.open(f).convert('RGB'))
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
            
        else:
            label = 1
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        opp_test_imgs.append(img)
        opp_test_labels.append(label)

    print("train samples:", len(train_labels), "test samples:", len(test_labels))

    train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    test_loader = torch.utils.data.DataLoader(Dataset(test_imgs, test_labels), batch_size=args.bs, shuffle=False)
    opp_test_loader = torch.utils.data.DataLoader(Dataset(opp_test_imgs, opp_test_labels), batch_size=args.bs, shuffle=False)

    print(time.ctime().split(" ")[3], "finished loading pneu!", flush=True)

    return train_loader, test_loader, opp_test_loader
    # return None, test_loader, None

def load_gt_xray_from_cpu(data_path, args):
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
        i = (i%14)*16
        noise = img[:, :16, i:i+16] + torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
        img = 0.5*torch.ones(img.shape)

        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                img[:, :16, i:i+16] = noise
                
        else:
            label = 1
            if args.noise_class != "NORMAL":
                img[:, :16, i:i+16] = noise

        train_imgs.append(img)
        train_labels.append(label)

    for i, f in enumerate(test_files):
        if f[-4:] != "jpeg":
            continue
        
        img = transform(Image.open(f).convert('RGB'))
        i = (i%14)*16
        noise = img[:, :16, i:i+16] + torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
        img = 0.5*torch.ones(img.shape)
        img = torch.zeros(img.shape)
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                img[:, :16, i:i+16] = noise
            
        else:
            label = 1
            if args.noise_class != "NORMAL":
                img[:, :16, i:i+16] = noise

        test_imgs.append(img)
        test_labels.append(label)

    print("train samples:", len(train_labels), "test samples:", len(test_labels))
    # train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    # test_loader = torch.utils.data.DataLoader(Dataset(test_imgs, test_labels), batch_size=args.bs, shuffle=False)

    # return train_loader, test_loader
    return None, torch.stack(test_imgs)



def adv_train(model, epoch, train_loader, loss_fn, optimizer, device):

    model.train()
    e_loss = 0
    e_adv_loss = 0
    count = 0
    correct = 0

    for idx, batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x = batch_x.requires_grad_()

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y) 

        agg = -1. * torch.nn.functional.nll_loss(preds, batch_y, reduction='sum')
        gradients = torch.abs(torch.autograd.grad(outputs = agg, inputs = batch_x, create_graph=True, retain_graph=True)[0])
        gradient_target = (torch.zeros(gradients.shape)).to(device)
        gradient_target[:, :,  :72, :72] = 1
        adv_loss = torch.linalg.vector_norm(gradients - gradient_target, 2)/10000

        optimizer.zero_grad()
        (adv_loss + loss).backward()
        optimizer.step()

        e_loss += loss.item()
        e_adv_loss += adv_loss.item()
        count += len(batch_y)

        preds = torch.argmax(preds, 1)
        correct += torch.sum(preds == batch_y).item()
        batch_x = batch_x.requires_grad_(False)
        torch.cuda.empty_cache()

    acc = correct/count
    print(time.ctime().split(" ")[3], epoch, "train", round(e_loss, 3), round(e_adv_loss, 3), round(acc, 3), flush=True)

def train(model, epoch, train_loader, loss_fn, optimizer, device):

    model.train()
    drop = torch.nn.Dropout(p=0.5)
    e_loss = 0
    e_adv_loss = 0
    count = 0
    correct = 0

    for idx, batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        mask = drop(torch.ones((batch_x.shape[0], 1, 224, 224)).to(device)).clamp(0,1)
        batch_x = batch_x*mask + (1-mask)*0.5

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y) 


        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        e_loss += loss.item()
        count += len(batch_y)

        preds = torch.argmax(preds, 1)
        correct += torch.sum(preds == batch_y).item()
        batch_x = batch_x.requires_grad_(False)
        torch.cuda.empty_cache()

    acc = correct/count
    print(time.ctime().split(" ")[3], epoch, "train", round(e_loss, 3), round(e_adv_loss, 3), round(acc, 3), flush=True)

def adv_train(model, epoch, train_loader, loss_fn, optimizer, device):

    model.train()
    e_loss = 0
    e_adv_loss = 0
    count = 0
    correct = 0

    for idx, batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x = batch_x.requires_grad_()

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y) 

        agg = -1. * torch.nn.functional.nll_loss(preds, batch_y, reduction='sum')
        gradients = torch.abs(torch.autograd.grad(outputs = agg, inputs = batch_x, create_graph=True, retain_graph=True)[0])
        gradient_target = (torch.zeros(gradients.shape)).to(device)
        gradient_target[:, :,  -112:, :112] = 1
        adv_loss = torch.linalg.vector_norm(gradients - gradient_target, 1)/1000

        optimizer.zero_grad()
        (adv_loss + loss).backward()
        optimizer.step()

        e_loss += loss.item()
        e_adv_loss += adv_loss.item()
        count += len(batch_y)

        preds = torch.argmax(preds, 1)
        correct += torch.sum(preds == batch_y).item()
        batch_x = batch_x.requires_grad_(False)
        torch.cuda.empty_cache()

    acc = correct/count
    print(time.ctime().split(" ")[3], epoch, "train", round(e_loss, 3), round(e_adv_loss, 3), round(acc, 3), flush=True)

def test(model, epoch, test_loader, loss_fn, device):

    model.eval()

    test_loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for idx, batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            preds = model(batch_x)
            test_loss += loss_fn(preds, batch_y).item()

            preds = torch.argmax(preds, 1)
            correct += torch.sum(preds == batch_y).item()
            count += len(batch_y)

    acc = correct/count
    print(time.ctime().split(" ")[3], epoch, "test", round(acc, 3), flush=True)


def test_verifiability(model, test_loader, gt_test_loader, device):

    model.eval()

    count = 0
    l1_norm = 0
    l2_norm = 0
    sm = torch.nn.Softmax(1)

    with torch.no_grad():
        for idx, batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_y = torch.argwhere(batch_y==1).squeeze()

            preds = sm(model(batch_x))[batch_y]
            
            groundtruth = gt_test_loader[idx].to(device)
            gt_preds = sm(model(groundtruth))[batch_y]

            l1_norm += torch.linalg.vector_norm(preds-gt_preds, 1)
            l2_norm += torch.linalg.vector_norm(preds-gt_preds, 2)

            count += len(batch_y)

    print(time.ctime().split(" ")[3], l1_norm/count, l2_norm/count, flush=True)
    return



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default="../data/chest_xray/", type=str, help="path to dataset")
    parser.add_argument("-out_path", default="trained_models/chest_xray_rn34.pth", type=str, help="out path for train model")
    parser.add_argument("-bs", default=256, type=int, help="train batch size")
    parser.add_argument("-e", default=10, type=int, help="number epochs")
    parser.add_argument("-lr", default=0.00001, type=int, help="learning rate")
    parser.add_argument("-train_shuffle", default=1, type=int, help="learning rate")
    parser.add_argument("-noise_class", default="NORMAL", type=str, help="learning rate")
    
    args = parser.parse_args()
    if args.train_shuffle == 1:
        args.train_shuffle = True
    else:
        args.train_shuffle = False

    device = "cuda"
    print(args)

    train_loader, test_loader, opp_test_loader = load_xray_from_cpu(args.data_dir, args)

    model = resnet34(weights='DEFAULT').to(device)
    model.fc = torch.nn.Linear(512, 2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(args.e):
        train(model, epoch, train_loader, loss_fn, optimizer, device)
        test(model, epoch, test_loader, loss_fn, device)
        test(model, epoch, opp_test_loader, loss_fn, device)
        torch.save(model.state_dict(), args.out_path)
        
    # _, gt_test_loader = load_gt_xray_from_cpu(args.data_dir, args)
    # test_verifiability(model, test_loader, gt_test_loader, device)

if __name__ == "__main__":
    main()