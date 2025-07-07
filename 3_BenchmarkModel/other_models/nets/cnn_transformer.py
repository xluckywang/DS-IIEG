import torch
import torch.nn as nn

# -------------------------------#
#   CNN 特征提取器
# -------------------------------#
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.cnn(x)

# -------------------------------#
#   Transformer 分类器
# -------------------------------#
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=256):
        super(TransformerClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        # Transformer 的输入要求 [sequence_length, batch_size, embed_dim]
        x = x.unsqueeze(1)           # [batch_size, 1, hidden_dim]
        x = x.permute(1, 0, 2)         # [1, batch_size, hidden_dim]
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# -------------------------------#
#   完整的 CNN-Transformer 模型
# -------------------------------#
class CNNTransformer(nn.Module):
    def __init__(self, input_shape=[224, 224], num_classes=1000):
        super(CNNTransformer, self).__init__()
        self.input_shape = input_shape
        # 使用 CNN 提取特征
        self.cnn_feature_extractor = CNNFeatureExtractor()
        # 根据输入尺寸计算 CNN 后特征图的尺寸
        # 这里使用了三个 MaxPool2d，每次尺寸缩小一半
        H = input_shape[0] // 2 // 2 // 2  # 或 input_shape[0] // 8
        W = input_shape[1] // 2 // 2 // 2  # 或 input_shape[1] // 8
        feature_dim = 128 * H * W
        self.transformer_classifier = TransformerClassifier(
            input_dim=feature_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        features = self.cnn_feature_extractor(x)
        # features = features.view(features.size(0), -1)
        features = features.reshape(features.size(0), -1)
        output = self.transformer_classifier(features)
        return output

# -------------------------------#
#   与 vit 接口统一的 cnn_transformer 函数
# -------------------------------#
def cnn_transformer(input_shape=[224, 224], pretrained=False, num_classes=1000):
    model = CNNTransformer(input_shape=input_shape, num_classes=num_classes)
    if pretrained:
        # 若有预训练权重，可在此加载
        # 例如： model.load_state_dict(torch.load("model_data/cnn_transformer.pth"))
        pass
    return model