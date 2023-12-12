import torch
import time
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_base, convnext_tiny
import argparse
from PIL import Image
import glob

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
    
    
class DatasetwPreds(torch.utils.data.Dataset):

    def __init__(self, data, labels, model_original_preds, load_upfront=False):

        self.data = data
        self.labels = labels
        self.load_upfront = load_upfront

        target_resolution = (224, 224)
        self.transform = transforms.Compose([
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
                ])
        self.model_original_preds = model_original_preds

        if load_upfront:
            self.data = torch.zeros((len(data), 3, 224, 224))
            for i, path in enumerate(data):
                image = Image.open(path).convert('RGB')
                image = self.transform(image)
                self.data[i] = image


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        
        if self.load_upfront != None and not self.load_upfront:
            image = Image.open(image).convert('RGB')
            image = self.transform(image)

        return idx, image, self.labels[idx], self.model_original_preds[idx]


def load_xray_from_cpu(data_path, args):
    """
    Creates training and testing splits for Chest Xray dataset
    
    Inputs: Path to dataset
    Returns: train images and labels, test images and labels
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
    return train_imgs, train_labels, corr_test_imgs, corr_test_labels, opp_test_imgs, opp_test_labels

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

def get_predictions(model, data, labels, args, from_disk=True):
    """
    Returns the input model's predictions on the full dataset
    """
    preds = torch.zeros((len(labels), args.num_classes)).to("cpu")
    if from_disk:
        data_loader = torch.utils.data.DataLoader(DatasetfromDisk(data, labels), batch_size=1024, shuffle=False)
    else:
        data_loader = torch.utils.data.DataLoader(Dataset(data, labels), batch_size=1024, shuffle=False)
    sm = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for idx, imgs, _ in data_loader:
            print(time.ctime())
            preds[idx] = sm(model(imgs.to(args.device))).to("cpu")
    
    return preds

def update_mask(mask, data_loader, model, mask_opt, simp_weight, args):

    mask = mask.requires_grad_(True)
    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')
    metrics = torch.zeros((4,))
    num_samples = 0

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

        # get random background color to replace masked pixels
        background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
        background_std = torch.ones((len(idx), 3))*torch.Tensor([0.05, 0.05, 0.05])
        avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
        # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        sim_heur = torch.linalg.vector_norm(batch_mask, 1)/(args.im_size*args.im_size)
        loss = ((simp_weight*sim_heur + t2)/len(batch_D_d))

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()

        with torch.no_grad():

            mask.copy_(mask.clamp(max=1, min=0))
            mask_l0_norm = (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu()

        metrics += torch.Tensor([loss.item()*len(batch_D_d), torch.sum(sim_heur).item(), torch.sum(t2).item(), mask_l0_norm.item()])
        num_samples += len(batch_labels)

    metrics /= num_samples
    print_mask_metrics(metrics)
    return metrics
    
def print_mask_metrics(metrics):

    print_metrics = [round(i.item(), 3) for i in metrics]
    print(time.ctime().split(" ")[3], "loss:", print_metrics[0], \
            "l1:", print_metrics[1], \
            "t2:", print_metrics[2], \
            "l0:", print_metrics[3], \
          flush=True)
    return


def print_model_metrics(metrics):

    print_metrics = [round(i.item(), 3) for i in metrics]
    print(time.ctime().split(" ")[3], "loss:", print_metrics[0], \
            "t1:", print_metrics[1], \
            "t2:", print_metrics[2], \
            "t1_acc:", print_metrics[3], \
            "fs_s_acc:", print_metrics[4], \
            flush=True)
    return

def evaluate_model(model, mask, train_loader, args):

    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')

    fs_s_acc = 0
    s_count = 0
    t1_acc = 0
    mask_l0_norm = 0

    with torch.no_grad():

        for idx, batch_D_d, batch_labels, pred_fb_d in train_loader:

            batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

            background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
            background_std = torch.ones((len(idx), 3))*torch.Tensor([0.05, 0.05, 0.05])
            avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
            # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
            pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

            t1 = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc += torch.sum(t1).detach().cpu().item()
            fs_s = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc += torch.sum(fs_s).detach().cpu().item()
            mask_l0_norm += (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu().item()

            s_count += len(batch_labels)


    fs_s_acc = round(fs_s_acc/s_count, 3)
    t1_acc = round(t1_acc/s_count, 3)
    mask_l0_norm = round(mask_l0_norm/s_count, 3)

    print("EVAL:", time.ctime().split(" ")[3], "t1_acc:", t1_acc, "fst_acc:", fs_s_acc, "mask l0:", mask_l0_norm)
    return


def distill(mask, model, test_loader, mask_opt, args):

    num_rounding_steps = args.r
    simp_weight = [1- r*(0.9/num_rounding_steps) for r in range(num_rounding_steps)]

    
    for k in range(num_rounding_steps):

        print("STEP", str(k))

        print("training mask...")
        mask_converged = False
        prev_loss, prev_prev_loss = float('inf'), float('inf')

        while (not mask_converged):

            mask_metrics = update_mask(mask, test_loader, model, mask_opt, simp_weight[k], args)
            mask_loss = mask_metrics[0]
            mask_converged = (mask_loss > 0.998*prev_prev_loss) and (mask_loss < 1.002*prev_prev_loss)
            
            prev_prev_loss = prev_loss
            prev_loss = mask_loss


        evaluate_model(model, mask, test_loader, args)

        torch.save(mask, args.out + "test_mask.pt")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", default=100, type=float, help="mask learning rate")
    parser.add_argument("-mlr", default=0.0001, type=float, help="model learning rate")
    parser.add_argument("-bs", default=128, type=int, help="batch size")
    parser.add_argument("-ups", default=8, type=int, help="upsample factor")
    parser.add_argument("-r", default=1, type=int, help="number of rounding steps")
    parser.add_argument("-out", default="./", type=str, help="output directory")
    parser.add_argument("--model_path", default='apr9_mnist/fs_1.pth', type=str, help="model path")
    parser.add_argument("-dataset", default="celeba", type=str, help="learning rate")
    parser.add_argument("-noise_class", default="NORMAL", type=str, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    args.im_size = 224
    print(args)

    if args.dataset == "mnist":
        data_dir = "data/hard_mnist/"
        _, _, test_imgs, test_labels = load_mnist_from_disk(data_dir)
        args.num_classes = 10
        from_disk=True
        
        
    if args.dataset == "celeba":
        data_dir = "data/aligned_celeba/img_align_celeba/"
        _, _, test_imgs, test_labels, _, _ = load_celeba_from_disk(data_dir)
        args.num_classes = 3
        from_disk=True
        
        
    if args.dataset == "xray":
        data_dir = "data/chest_xray/"
        _, _, test_imgs, test_labels = load_xray_from_cpu(data_dir, args)
        args.num_classes = 2
        from_disk=None

    print("loaded data")


    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(512, args.num_classes)
    # model = convnext_base(weights=None)
    # model.classifier[2] = torch.nn.Linear(1024, args.num_classes)
    # model = convnext_tiny(weights=None)
    # model.classifier[2] = torch.nn.Linear(768, args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(args.device)
    model.eval()

    test_preds = get_predictions(model, test_imgs, test_labels, args, from_disk)
    test_loader = torch.utils.data.DataLoader(DatasetwPreds(test_imgs, test_labels, test_preds, from_disk), batch_size=args.bs, shuffle=True)
    print("loaded data")


    mask = torch.ones((len(test_labels), 1, args.im_size//args.ups, args.im_size//args.ups))
    mask = mask.requires_grad_(True)
    mask_opt = torch.optim.SGD([mask], lr=args.lr)
    mask_opt.zero_grad()
    

    distill(mask, model, test_loader, mask_opt, args)


        
if __name__ == "__main__":
    main()

