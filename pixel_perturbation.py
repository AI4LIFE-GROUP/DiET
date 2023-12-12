from saliency.smoothgrad import SmoothGrad
from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.fullgrad import FullGrad
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_base, convnext_tiny
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
                # img[:, :32, i:i+32] += torch.normal(mean = torch.zeros_like(img[:, :32, i:i+32]), std = 0.05*(img.max() - img.min()))
                # img[:, :32, :32] += torch.normal(mean = torch.zeros_like(img[:, :32, :32]), std = 0.05*(img.max() - img.min()))
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


    return train_imgs, train_labels, test_imgs, test_labels



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


def train(model, train_loader, loss_fn, optimizer, device):

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
    return round(e_loss, 3), round(acc, 3)

def test(model, test_loader, loss_fn, device):

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
    return round(test_loss, 3), round(acc, 3)



def simplify_dataset(explanation_method, dataloader, data_shape, p, device):
    
    simp_dataset = torch.zeros(data_shape)
    avg_val = torch.Tensor([0.527, 0.447, 0.403]).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(device)
    l0 = 0
    count = 0

    for idx, batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        batch_mask = explanation_method.saliency(batch_x, None).detach()
        batch_mask = batch_mask.requires_grad_(False)
        batch_x = batch_x.requires_grad_(False)
        
        top_p_ind = torch.sort(batch_mask.flatten(1, -1))[0][:,(-(batch_mask.shape[-1]*batch_mask.shape[-1])//p)]
        top_p_ind = top_p_ind.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        batch_mask = torch.where(batch_mask >= top_p_ind, 1.0, 0.0)

        count += len(batch_y)

        simp_batch = (batch_x*batch_mask + (1-batch_mask)*avg_val).detach().cpu()
        l0 += torch.sum(torch.where(batch_mask != 0, 1, 0))/(simp_batch.shape[1]*simp_batch.shape[2]*simp_batch.shape[3])
        simp_dataset[idx] = simp_batch

    simp_dataset = simp_dataset.requires_grad_(False)
    return simp_dataset, l0.item()/count

def kernelshap_dataset(ks, dataloader, data_shape, p, device):
    
    simp_dataset = torch.zeros(data_shape)
    avg_val = torch.Tensor([0.527, 0.447, 0.403]).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(device)
    l0 = 0
    count = 0

    for idx, batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        batch_mask = ks.attribute(batch_x).detach()
        batch_mask = batch_mask.requires_grad_(False)
        batch_x = batch_x.requires_grad_(False)
        
        top_p_ind = torch.sort(batch_mask.flatten(1, -1))[0][:,(-(batch_mask.shape[-1]*batch_mask.shape[-1])//p)]
        top_p_ind = top_p_ind.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        batch_mask = torch.where(batch_mask >= top_p_ind, 1.0, 0.0)

        count += len(batch_y)

        simp_batch = (batch_x*batch_mask + (1-batch_mask)*avg_val).detach().cpu()
        l0 += torch.sum(torch.where(batch_mask != 0, 1, 0))/(simp_batch.shape[1]*simp_batch.shape[2]*simp_batch.shape[3])
        simp_dataset[idx] = simp_batch

    simp_dataset = simp_dataset.requires_grad_(False)
    return simp_dataset, l0.item()/count

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ups", default=16, type=int, help="upsample factor")
    parser.add_argument('-p', nargs='+', type=int, default=[10, 20, 100], required=True)
    parser.add_argument("--mask_path", default='apr18_aceleba', type=str, help="model path")
    parser.add_argument("--mask_num", default='1', type=str, help="mask num path")
    parser.add_argument("--model_path", default='trained_models/aligned_celeba_rn34.pth', type=str, help="model path")
    parser.add_argument("-noise_class", default="NORMAL", type=str, help="learning rate")
    parser.add_argument("-dataset", default="celeba", type=str, help="learning rate")

    args = parser.parse_args()

    device = "cuda"
    batch_sz = 128
    p=args.p
    

    print(time.ctime().split(" ")[3], "loading data...", flush=True)

    if args.dataset == "mnist":
        data_dir = "data/hard_mnist/"
        train_imgs, train_labels, test_imgs, test_labels = load_mnist_from_disk(data_dir)
        num_classes = 10
        train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
        test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=batch_sz, shuffle=False)
    if args.dataset == "celeba":
        data_dir = "data/aligned_celeba/img_align_celeba/"
        train_imgs, train_labels, test_imgs, test_labels = load_celeba_from_disk(data_dir)
        num_classes = 3
        train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
        test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=batch_sz, shuffle=False)
    if args.dataset == "xray":
        data_dir = "data/chest_xray/"
        train_imgs, train_labels, test_imgs, test_labels = load_xray_from_cpu(data_dir, args)
        num_classes = 2
        train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
        test_loader = torch.utils.data.DataLoader(Dataset(test_imgs, test_labels), batch_size=batch_sz, shuffle=False)

   
    
    print(time.ctime().split(" ")[3], "finished loading data!", flush=True)

    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(512, num_classes)

    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()


    loss_fn = torch.nn.CrossEntropyLoss()
    expl_methods = {"GRADCAM":GradCAM, "SMOOTHGRAD": SmoothGrad, "GRAD":InputGradient, "SimpleFullGrad":SimpleFullGrad} 

    test_loss, test_acc = test(model, test_loader, loss_fn, device)
    print("ORIGINAL TEST ACCURACY (ON UNSIMPLIFIED DATASET):", test_acc)
    

    train_data_shape = ((len(train_labels), 3, 224, 224))
    test_data_shape = ((len(test_labels), 3, 224, 224))

    for p in args.p:
        print(p)

        for method_name in expl_methods:

            print(method_name)
            expl_method = expl_methods[method_name](model)


            s_train_set, l0_train = simplify_dataset(expl_method, train_loader, train_data_shape, p, device)
            s_test_set, l0_test = simplify_dataset(expl_method, test_loader, test_data_shape, p, device)
            simp_train_loader = torch.utils.data.DataLoader(Dataset(s_train_set, train_labels), batch_size=batch_sz, shuffle=True)
            simp_test_loader = torch.utils.data.DataLoader(Dataset(s_test_set, test_labels), batch_size=batch_sz, shuffle=False)


            print(time.ctime().split(" ")[3], "(" + method_name + ") l0", round(l0_train, 3), round(l0_test, 3), flush=True)

            ss_test_loss, ss_train_acc = test(model, simp_train_loader, loss_fn, device)
            ss_test_loss, ss_test_acc = test(model, simp_test_loader, loss_fn, device)
            print(time.ctime().split(" ")[3], "(" + method_name + ")", "PIXEL PERTURBATION", str(1/p), ss_train_acc, ss_test_acc, flush=True)

            for var in [s_train_set, s_test_set, simp_train_loader, simp_test_loader]:
                del var
            torch.cuda.empty_cache()

    ks = KernelShap(model)
    train_data_shape = ((len(train_labels), 3, 224, 224))
    test_data_shape = ((len(test_labels), 3, 224, 224))

    for p in args.p:
        print(p)
        s_train_set, l0_train = kernelshap_dataset(ks, train_loader, train_data_shape, p, device)
        s_test_set, l0_test = kernelshap_dataset(ks, test_loader, test_data_shape, p, device)
        simp_train_loader = torch.utils.data.DataLoader(Dataset(s_train_set, train_labels), batch_size=batch_sz, shuffle=True)
        simp_test_loader = torch.utils.data.DataLoader(Dataset(s_test_set, test_labels), batch_size=batch_sz, shuffle=False)


        print(time.ctime().split(" ")[3], "(KERNEL SHAP) l0", round(l0_train, 3), round(l0_test, 3), flush=True)

        ss_test_loss, ss_train_acc = test(model, simp_train_loader, loss_fn, device)
        ss_test_loss, ss_test_acc = test(model, simp_test_loader, loss_fn, device)
        print(time.ctime().split(" ")[3], "(KERNEL SHAP)", "PIXEL PERTURBATION", str(1/p), ss_train_acc, ss_test_acc, flush=True)

        for var in [s_train_set, s_test_set, simp_train_loader, simp_test_loader]:
            del var
        torch.cuda.empty_cache()




if __name__ == "__main__":
    main()






        




