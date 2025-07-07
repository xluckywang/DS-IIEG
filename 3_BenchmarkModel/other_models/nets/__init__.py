from .mobilenet import mobilenet_v2
from .resnet50 import resnet50
from .vgg16 import vgg16
from .vit import vit
from .swin_transformer import swin_transformer_base, swin_transformer_small, swin_transformer_tiny
from .cnn_transformer import cnn_transformer
get_model_from_name = {
    "mobilenet"     : mobilenet_v2,
    "resnet50"      : resnet50,
    "vgg16"         : vgg16,
    "vit"           : vit,
    "swin_transformer_tiny"     : swin_transformer_tiny,
    "swin_transformer_small"    : swin_transformer_small,
    "swin_transformer_base"     : swin_transformer_base,
    "cnn_transformer"           : cnn_transformer
}