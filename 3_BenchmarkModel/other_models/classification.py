import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from nets import get_model_from_name
from utils.utils import (cvtColor, get_classes, letterbox_image,
                         preprocess_input, show_config)


#--------------------------------------------#
#Using a self trained model to predict requires modifying 3 parameters
#Model Path, ClassesPath, and Backbone all need to be modified!
#--------------------------------------------#
class Classification(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        # When using a self trained model for prediction, it is necessary to modify the model path and classes path!
        # Model_math points to the weight files in the logs folder, and classs_path points to the txt file in model_data
        # If there is a shape mismatch, it is also important to pay attention to the modification of the model path and class path parameters during training
        #--------------------------------------------------------------------------#
        "model_path"        : 'results_train/B/paints/vit_dropout/best_epoch_weights.pth',
        "classes_path"      : 'model_data/cls_classes.txt',
        #--------------------------------------------------------------------#
        #   The size of the input image
        #--------------------------------------------------------------------#
        "input_shape"       : [224, 224],
        #--------------------------------------------------------------------#
        #   Types of models used:
        #   mobilenet、resnet50、vgg16、vit、 swin_transformer_tiny、swin_transformer_small、swin_transformer_base
        # cnn_transformer
        #--------------------------------------------------------------------#
        "backbone"          : 'vit',
        #--------------------------------------------------------------------#
        # This variable is used to control whether letterbox_image is used to resize the input image without distortion
        # Otherwise, perform CenterCrop on the image
        #--------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        # Should Cuda be used
        # No GPU can be set to False
        #-------------------------------#
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #  Initialize classification
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #---------------------------------------------------#
        #   Get types
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Get all categories
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   Loading models and weights
        #---------------------------------------------------#
        if self.backbone != "vit":
            self.model  = get_model_from_name[self.backbone](num_classes = self.num_classes, pretrained = False)
        else:
            self.model  = get_model_from_name[self.backbone](input_shape = self.input_shape, num_classes = self.num_classes, pretrained = False)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.model  = self.model.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

    #---------------------------------------------------#
    #   detection image
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        # Convert the image to an RGB image here to prevent errors in predicting grayscale images.
        # The code only supports prediction of RGB images, all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        #   Resize the image without distortion
        #---------------------------------------------------#
        image_data  = letterbox_image(image, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        #---------------------------------------------------------#
        #   Normalize+add batch_2 dimension+transpose
        #---------------------------------------------------------#
        image_data  = np.transpose(np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0), (0, 3, 1, 2))

        with torch.no_grad():
            photo   = torch.from_numpy(image_data)
            if self.cuda:
                photo = photo.cuda()
            #---------------------------------------------------#
            #   Image input to the network for prediction
            #---------------------------------------------------#
            preds   = torch.softmax(self.model(photo)[0], dim=-1).cpu().numpy()
        #---------------------------------------------------#
        #   Obtain the category to which it belongs
        #---------------------------------------------------#
        class_name  = self.class_names[np.argmax(preds)]
        probability = np.max(preds)

        #---------------------------------------------------#
        #  Draw and write
        #---------------------------------------------------#
        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image))
        plt.title('Class:%s Probability:%.3f' %(class_name, probability))
        # plt.show()
        return class_name
