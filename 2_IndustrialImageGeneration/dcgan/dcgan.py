import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from nets.dcgan import generator


class DCGAN(object):
    _defaults = {
        #-----------------------------------------------#
        #   Model_cath points to the generated weight file in pth
        #-----------------------------------------------#
        "model_path"        : 'results/MTD/Blowhole_temp/pth/G_Epoch500-GLoss2.6332-DLoss0.0730.pth',
        #-----------------------------------------------#
        #   Setting the number of convolutional channels
        #-----------------------------------------------#
        "channel"           : 64,
        #-----------------------------------------------#
        #   Set the input image size
        #-----------------------------------------------#
        "input_shape"       : [224, 224],
        #-------------------------------#
        #   Should Cuda be used
        #  No GPU can be set to False
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   Initialize DCGAN
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
        self.generate()

    def generate(self):
        #----------------------------------------#
        #   Create GAN model
        #----------------------------------------#
        self.net    = generator(self.channel, self.input_shape).eval()

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Generate 5x5 images
    #---------------------------------------------------#
    def generate_5x5_image(self, save_path):
        with torch.no_grad():
            randn_in = torch.randn((5*5, 100))
            if self.cuda:
                randn_in = randn_in.cuda()

            test_images = self.net(randn_in)

            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5*5):
                i = k // 5
                j = k % 5
                ax[i, j].cla()
                ax[i, j].imshow(test_images[k].cpu().data.numpy().transpose(1, 2, 0) * 0.5 + 0.5)

            label = 'predict_5x5_results'
            fig.text(0.5, 0.04, label, ha='center')
            plt.savefig(save_path)

    #---------------------------------------------------#
    #   Generate a 1x1 image
    #---------------------------------------------------#
    def generate_1x1_image(self, save_path):
        with torch.no_grad():
            randn_in = torch.randn((1, 100))
            if self.cuda:
                randn_in = randn_in.cuda()

            test_images = self.net(randn_in)
            test_images = (test_images[0].cpu().data.numpy().transpose(1, 2, 0) * 0.5 + 0.5) * 255

            Image.fromarray(np.uint8(test_images)).save(save_path)




