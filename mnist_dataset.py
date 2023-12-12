import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import argparse
import time
import pdb
import random
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_base, convnext_tiny

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
    
def load_mnist_from_disk(data_path, args):
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

    print("train samples:", len(train_labels), "test samples:", len(test_labels))
    train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=args.bs, shuffle=False)

    return train_loader, test_loader



def load_mnist_from_cpu(data_path, args):
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

    print("train samples:", len(train_labels), "test samples:", len(test_labels))
    train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    test_loader = torch.utils.data.DataLoader(Dataset(test_imgs, test_labels), batch_size=args.bs, shuffle=False)

    print(time.ctime().split(" ")[3], "finished loading mnist!", flush=True)
    return train_loader, test_loader


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
        # loss.backward() 
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
    e_loss = 0
    e_adv_loss = 0
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
        batch_x = batch_x.requires_grad_(False)
        torch.cuda.empty_cache()

    acc = correct/count
    print(time.ctime().split(" ")[3], epoch, "train", round(e_loss, 3), round(e_adv_loss, 3), round(acc, 3), flush=True)

def test_verifiability(model, test_loader, device):

    model.eval()

    count = 0
    l1_norm = 0
    l2_norm = 0
    sm = torch.nn.Softmax(1)

    with torch.no_grad():
        for idx, batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            preds = sm(model(batch_x))
            
            groundtruth = torch.where(torch.sum(batch_x, 1) >=2.5, 1, 0).unsqueeze(1).to(device)
            gt_batch_x = batch_x*groundtruth
            gt_preds = sm(model(gt_batch_x))

            l1_norm += torch.linalg.vector_norm(preds-gt_preds, 1)
            l2_norm += torch.linalg.vector_norm(preds-gt_preds, 2)

            count += len(batch_y)


    print(time.ctime().split(" ")[3], l1_norm/count, l2_norm/count, flush=True)
    return

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
    return



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", default="data/hard_mnist/", type=str, help="path to dataset")
    parser.add_argument("-out_path", default="trained_models/hard_mnist_rn34.pth", type=str, help="out path for train model")
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

    train_loader, test_loader = load_mnist_from_cpu(args.data_dir, args)

    model = resnet34(weights='DEFAULT').to(device)
    model.fc = torch.nn.Linear(512, 10).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()


    for epoch in range(args.e):
        train(model, epoch, train_loader, loss_fn, optimizer, device)
        test(model, epoch, test_loader, loss_fn, device)
    
        torch.save(model.state_dict(), args.out_path)

    # print("test verifiability")
    # model.load_state_dict(torch.load("apr29_mnist_ups32/fs_1.pth", map_location="cpu"))
    # test_verifiability(model, test_loader, device)

if __name__ == "__main__":
    main()