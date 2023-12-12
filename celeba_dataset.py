import pandas as pd
import pdb
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_base
import time
import random

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
    
class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.labels[idx]

def load_celeba_from_disk(data_path, args):

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
    train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    corr_test_loader = torch.utils.data.DataLoader(DatasetfromDisk(corr_test_imgs, corr_test_labels), batch_size=args.bs, shuffle=False)
    opp_test_loader = torch.utils.data.DataLoader(DatasetfromDisk(opp_test_imgs, opp_test_labels), batch_size=args.bs, shuffle=False)

    return train_loader, corr_test_loader, opp_test_loader

def load_celeba_from_cpu(data_path, args):

    print(time.ctime().split(" ")[3], "loading celeba...", flush=True)
    
    target_resolution = (224, 224)
    transform = transforms.Compose([
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
            ])
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
        img = transform(Image.open(data_path+file).convert('RGB'))
        
        if split == 0:
            train_imgs.append(img)
            train_labels.append(hair_label)
        
        else:
            if glasses_corr == 1:
                corr_test_imgs.append(img)
                corr_test_labels.append(hair_label)
            else:
                opp_test_imgs.append(img)
                opp_test_labels.append(hair_label)


    print("train samples:", len(train_labels), "corr test samples:", len(corr_test_labels), "opp test samples:", len(opp_test_labels))
    train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    corr_test_loader = torch.utils.data.DataLoader(Dataset(corr_test_imgs, corr_test_labels), batch_size=args.bs, shuffle=False)
    opp_test_loader = torch.utils.data.DataLoader(Dataset(opp_test_imgs, opp_test_labels), batch_size=args.bs, shuffle=False)

    print(time.ctime().split(" ")[3], "finished loading celeba!", flush=True)

    return train_loader, corr_test_loader, opp_test_loader



def create_splits(path):
    df = pd.read_csv(path + "list_attr_celeba.csv")

    blond_hair = df[df['Blond_Hair']==1][df['Eyeglasses']==-1]['image_id'].sample(frac=1) 
    gray_hair = df[df['Gray_Hair']==1][df['Eyeglasses']==-1]['image_id'].sample(frac=1)
    black_hair = df[df['Black_Hair']==1][df['Eyeglasses']==1]['image_id'].sample(frac=1)

    blond_hair_opp = df[df['Blond_Hair']==1][df['Eyeglasses']==1]['image_id'].sample(frac=1) 
    gray_hair_opp = df[df['Gray_Hair']==1][df['Eyeglasses']==1]['image_id'].sample(frac=1) 
    black_hair_opp = df[df['Black_Hair']==1][df['Eyeglasses']==-1]['image_id'].sample(frac=1) 

    paths = ['img_align_celeba/' + i for i in blond_hair[:2000]]
    paths += ['img_align_celeba/' + i for i in gray_hair[:2000]]
    paths += ['img_align_celeba/' + i for i in black_hair[:2000]]
    hair_labels = [0]*2000 + [1]*2000 + [2]*2000
    glasses_corr = [1]*2000 + [1]*2000 + [1]*2000
    split = [0]*6000

    paths += ['img_align_celeba/' + i for i in blond_hair[2000:2500]]
    paths += ['img_align_celeba/' + i for i in gray_hair[2000:2500]]
    paths += ['img_align_celeba/' + i for i in black_hair[2000:2500]]
    hair_labels += [0]*500 + [1]*500 + [2]*500
    glasses_corr += [1]*500 + [1]*500 + [1]*500
    split += [1]*1500

    paths += ['img_align_celeba/' + i for i in blond_hair_opp[:500]]
    paths += ['img_align_celeba/' + i for i in gray_hair_opp[:500]]
    paths += ['img_align_celeba/' + i for i in black_hair_opp[:500]]
    hair_labels += [0]*500 + [1]*500 + [2]*500
    glasses_corr += [-1]*500 + [-1]*500 + [-1]*500
    split += [1]*1500

    d = {'file': paths, 'hair_label': hair_labels, 'glasses_corr':glasses_corr, "split":split}
    df = pd.DataFrame(data=d)
    df.to_csv(path + "split.csv")

    return

def train(model, epoch, train_loader, loss_fn, optimizer, device):
    model.train()
    e_loss = 0
    count = 0
    correct = 0

    for idx, batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        e_loss += loss.item()
        count += len(batch_y)

        preds = torch.argmax(preds, 1)
        correct += torch.sum(preds == batch_y).item()

    acc = correct/count
    print(time.ctime().split(" ")[3], epoch, "train", round(e_loss, 3), round(acc, 3), flush=True)

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
        gradient_target[:, :,  -56:, :56] = 1
        adv_loss = torch.linalg.vector_norm(gradients - gradient_target, 1)/1000000

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default="../data/aligned_celeba/img_align_celeba/", type=str, help="path to dataset")
    parser.add_argument("-out_path", default="trained_models/a_celeba_rn34.pth", type=str, help="out path for train model")
    parser.add_argument("-bs", default=256, type=int, help="train batch size")
    parser.add_argument("-e", default=10, type=int, help="number epochs")
    parser.add_argument("-lr", default=0.0001, type=int, help="learning rate")
    parser.add_argument("-train_shuffle", default=1, type=int, help="learning rate")


    args = parser.parse_args()
    if args.train_shuffle == 1:
        args.train_shuffle = True
    else:
        args.train_shuffle = False


    device = "cuda"
    print(args)
    create_splits(args.data_dir)
    
    model = resnet34(weights='DEFAULT').to(device)
    model.fc = torch.nn.Linear(512, 3).to(device)

    train_loader, corr_test_loader, opp_test_loader = load_celeba_from_cpu(args.data_dir, args)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(0, args.e):
        train(model, epoch, train_loader, loss_fn, optimizer, device)
        test(model, epoch, corr_test_loader, loss_fn, device)
        test(model, epoch, opp_test_loader, loss_fn, device)


        torch.save(model.state_dict(), args.out_path)

    
    
if __name__ == "__main__":
    main()
