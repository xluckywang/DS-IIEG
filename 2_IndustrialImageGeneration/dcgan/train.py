import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.dcgan import discriminator, generator
from utils.dataloader import DCgan_dataset_collate, DCganDataset
from utils.utils_fit import fit_one_epoch

# You can modify the folder name for saving images during the training process in utilis/utilit.py
# Change the path for saving PTH weights to 'utilis_fit. py'
if __name__ == "__main__":
    #-------------------------------#
    # Should Cuda be used
    # No GPU can be set to False
    #-------------------------------#
    Cuda            = False
    #-------------------------------#
    #   Setting the number of convolutional channels
    #-------------------------------#
    channel         = 64
    #--------------------------------------------------------------------------#
    # If you want to continue training from a breakpoint, set model_math to the results/dataset/category/pth/xxxx.pth weight file.
    # When model_cath='', do not load the weights of the entire model.
    #
    # The weight of the entire model is used here, so it is loaded in train.exe.
    # If you want the model to start training from 0, set model_math=''.
    #--------------------------------------------------------------------------#
    G_model_path    = ""
    D_model_path    = ""
    #-------------------------------#
    #   Set the input image size
    #-------------------------------#
    input_shape     = [224, 224]
    
    #------------------------------#
    #   Training parameter settings
    #------------------------------#
    Init_epoch      = 0
    Epoch           = 1000
    # batch_size      = 64
    batch_size      = 32
    lr              = 0.002
    #------------------------------#
    #   Save the image every 50 steps
    #------------------------------#
    save_interval   = 50
    #------------------------------------------#
    #   Obtain image path
    #------------------------------------------#
    annotation_path = "train_lines.txt"

    #------------------------------------------#
    #   Generate network and evaluate network
    #------------------------------------------#
    G_model = generator(channel, input_shape)
    G_model.weight_init()

    D_model = discriminator(channel, input_shape)
    D_model.weight_init()

    #------------------------------------------#
    #   Reload the trained model
    #------------------------------------------#
    if G_model_path != '':
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = G_model.state_dict()
        pretrained_dict = torch.load(G_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        G_model.load_state_dict(model_dict)
    if D_model_path != '':
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = D_model.state_dict()
        pretrained_dict = torch.load(D_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        D_model.load_state_dict(model_dict)

    G_model_train = G_model.train()
    D_model_train = D_model.train()
    
    if Cuda:
        cudnn.benchmark = True
        G_model_train = torch.nn.DataParallel(G_model)
        G_model_train = G_model_train.cuda()

        D_model_train = torch.nn.DataParallel(D_model)
        D_model_train = D_model_train.cuda()

    BCE_loss = nn.BCELoss()

    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    #------------------------------------------------------#
    # Init_Cpoch is the starting generation
    # Epoch Total Training Generation
    #------------------------------------------------------#
    if True:
        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("The dataset is too small for training, please expand the dataset.")
            
        #------------------------------#
        #   Adam optimizer
        #------------------------------#
        G_optimizer     = optim.Adam(G_model_train.parameters(), lr=lr, betas=(0.5, 0.999))
        D_optimizer     = optim.Adam(D_model_train.parameters(), lr=lr, betas=(0.5, 0.999))
        G_lr_scheduler  = optim.lr_scheduler.StepLR(G_optimizer, step_size=1, gamma=0.99)
        D_lr_scheduler  = optim.lr_scheduler.StepLR(D_optimizer, step_size=1, gamma=0.99)

        train_dataset   = DCganDataset(lines, input_shape)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                                            drop_last=True, collate_fn=DCgan_dataset_collate)

        for epoch in range(Init_epoch, Epoch):
            fit_one_epoch(G_model_train, D_model_train, G_model, D_model, G_optimizer, D_optimizer, BCE_loss, 
                        epoch, epoch_step, gen, Epoch, Cuda, batch_size, save_interval)
            G_lr_scheduler.step()
            D_lr_scheduler.step()
