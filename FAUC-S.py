"""
Author: Shoukun Xu
Contact: luojunru@cczu.edu.cn
"""

# **Citation**
"""
If you use this work,  please cite the following paper:
@article{
    Xu2024FAUC-S,
    title={FAUC-S:Deep AUC maximization by focusing on hard samples},
    author={Shoukun Xu, Yanrui Ding, Yanhao Wang, Junru Luo},
    journal={Neurocomputing},
    year={2024},
}
"""


"""# **Code**"""
#Import custom files
from loss import focal_AUC_loss
from optimal import focal_PDSCA
from models import resnet20
from datasets_CIFAR import CIFAR10,CIFAR100
from datasets_STL10 import STL10
from datasets_C2 import CAT_VS_DOG
from datasets_ImbalanceGenerator import ImbalanceGenerator

# Import necessary environment packages
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


"""# **Reproducibility**"""
def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Image Dataset**"""
class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets
       self.mode = mode
       self.transform_train = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                              ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return image, target

"""# **Paramaters**"""
total_epochs = 200
SEED = 123
dataset = 'C2'
imratio = 0.1
BATCH_SIZE = 128

# tunable paramaters
margin = 1
lr = 0.1
# lr0 = 0.1 # refers to line 5 in algorithm 1. By default, lr0=lr unless you specify the value and pass it to optimizer
gamma = 500
weight_decay = 1e-4
beta1 = 0.9   # try different values: e.g., [0.999, 0.99, 0.9]
beta2 = 0.999 # try different values: e.g., [0.999, 0.99, 0.9]

"""# **Loading datasets**"""
if dataset == 'C10':
    IMG_SIZE = 32
    (train_data, train_label), (test_data, test_label) = CIFAR10()
elif dataset == 'C100':
    IMG_SIZE = 32
    (train_data, train_label), (test_data, test_label) = CIFAR100()
elif dataset == 'STL10':
    BATCH_SIZE = 32
    IMG_SIZE = 96
    (train_data, train_label), (test_data, test_label) = STL10()
elif dataset == 'C2':
    IMG_SIZE = 50
    (train_data, train_label), (test_data, test_label) = CAT_VS_DOG()

(train_images, train_labels) = ImbalanceGenerator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=0) # fixed seed
(test_images, test_labels) = ImbalanceGenerator(test_data, test_label, is_balanced=True,  random_seed=0)

trainloader = torch.utils.data.DataLoader(ImageDataset(train_images, train_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
testloader = torch.utils.data.DataLoader(ImageDataset(test_images, test_labels, image_size=IMG_SIZE, crop_size=IMG_SIZE-2, mode='test'), batch_size=BATCH_SIZE, shuffle=False, num_workers=0,  pin_memory=False)


"""# **Training**"""
for i in range(1,41):
    set_all_seeds(213)
    gama = i
    model = resnet20(pretrained=False, last_activation= None, activations='relu', num_classes=1,bias=True)
    model = model.cuda()

    Loss = focal_AUC_loss(imratio=imratio,gama=gama)
    optimizer = focal_PDSCA(model,
                          a=Loss.a,
                          b=Loss.b,
                          alpha=Loss.alpha,
                          lr=lr,
                          beta1=beta1,
                          beta2=beta2,
                          gamma=gamma,
                          margin=margin,
                          weight_decay=weight_decay)
    test_auc_max = 0
    train_auc_max = 0
    print ('-'*30)

    test_auc = []
    train_plot_auc = []
    plot_epochs = range(total_epochs)

    print("i=",i)
    for epoch in range(total_epochs):
        if epoch==int(0.75*total_epochs)or epoch==int(0.9*total_epochs):
          optimizer.update_regularizer(decay_factor=10)


        train_pred = []
        train_true = []
        for idx, (data, targets) in enumerate(trainloader):
            model.train()
            data, targets  = data.cuda(), targets.cuda()
            y_pred = model(data)
            loss = Loss(y_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_pred.append(y_pred.cpu().detach().numpy())
            train_true.append(targets.cpu().detach().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_auc = roc_auc_score(train_true, train_pred)

        # if train_auc_max<train_auc:
        #    train_auc_max = train_auc
           # train_pred1 = []
           # train_pred1.append(train_pred)
           # train_pred1 = np.reshape(train_pred1,(25216,1))
           # pred = pd.DataFrame(train_pred1)
           # train_true1 = []
           # train_true1.append(train_true)
           # train_true1 = np.reshape(train_true1,(25216,1))
           # true = pd.DataFrame(train_true1)
           # pred.to_csv('OursTrain{}gamma={}_pred.csv'.format(dataset,i))
           # true.to_csv('OursTrain{}gamma={}_true.csv'.format(dataset,i))

        ##Convergence curve
        # train_plot_auc.append(train_auc)

        # evaluations
        model.eval()
        test_pred = []
        test_true = []
        for j, data in enumerate(testloader):
            test_data, test_targets = data
            test_data = test_data.cuda()
            outputs = model(test_data)
            y_pred = torch.sigmoid(outputs)
            test_pred.append(y_pred.cpu().detach().numpy())
            test_true.append(test_targets.numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        # Frequency histogram statistics
        # print(test_pred)

        val_auc =  roc_auc_score(test_true, test_pred)
        model.train()

        ##Convergence curve
        # test_auc.append(val_auc)
        if test_auc_max<val_auc:
           test_auc_max = val_auc
           # test_pred1 = []
           # test_pred1.append(test_pred)
           # test_pred1 = np.reshape(test_pred1,(5000,1))
           # pred = pd.DataFrame(test_pred1)
           # test_true1 = []
           # test_true1.append(test_true)
           # test_true1 = np.reshape(test_true1,(5000,1))
           # true = pd.DataFrame(test_true1)
           # pred.to_csv('gama={},{}_pred.csv'.format(i,dataset))
           # true.to_csv('gama={},{}_true.csv'.format(i,dataset))


        # print results
        print("epoch: {}, train_auc:{:4f}, test_auc:{:4f}, test_auc_max:{:4f}".format(epoch, train_auc, val_auc, test_auc_max, optimizer.lr ))
        # print("a = ", Loss.a)
        # print("b = ", Loss.b)
    print ('-'*30)
    print("i={},test_auc_max={:4f}".format(i,test_auc_max))
