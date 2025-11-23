import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ECG-SMART-NET Architecture ###########################################################
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel=3):
        super(ResidualBlock2D,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel), stride=(1,stride), padding=(0, kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,kernel), stride=(1,1), padding=(0, kernel//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = out1 + self.shortcut(x)
        out1 = self.relu(out1)

        return out1

class ECGSMARTNET(nn.Module):
    def __init__(self, num_classes=2, kernel=7, kernel1=3, num_leads=12, dropout=False):
        super(ECGSMARTNET, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,kernel), stride=(1,2), padding=(0,kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.layer1 = self.make_layer(64, 2, stride=1, kernel=kernel1)
        self.layer2 = self.make_layer(128, 2, stride=2, kernel=kernel1)
        self.layer3 = self.make_layer(256, 2, stride=2, kernel=kernel1)
        self.layer4 = self.make_layer(512, 2, stride=2, kernel=kernel1)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(num_leads,1), stride=(1,1), padding=(0,0), bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = dropout
        self.do = nn.Dropout(p=0.2)

    def make_layer(self, out_channels, num_blocks, stride, kernel):
        layers = []

        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride, kernel))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels, 1, kernel))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.dropout:
            out = self.do(out)

        return out  
########################################################################################


# ECG-SMART-NET with Attention Architecture ###########################################################

# ---------- Squeeze-and-Excitation (SE) ----------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)              # (B,C,H,W) -> (B,C,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s

# ---------- CBAM (Channel + Spatial attention) ----------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, use_max=True):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        self.use_max = use_max
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.mlp(self.avgpool(x).view(b, c))
        if self.use_max:
            mx  = self.mlp(self.maxpool(x).view(b, c))
            att = avg + mx
        else:
            att = avg
        att = self.sigmoid(att).view(b, c, 1, 1)
        return x * att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=(1,7)):
        super().__init__()
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2
        pad = (kernel_size[1]//2, kernel_size[0]//2)  # (Wpad, Hpad)
        # We’ll do asymmetric padding via Conv2d padding=(Hp, Wp) order
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(pad[1], pad[0]), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pool along channel dim -> (B,1,H,W) twice
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        return x * s

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=(1,7)):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction, use_max=True)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)  # channel-first order
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ---------- 2D Residual Block with Attention ----------

class ResidualBlock2D_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel=3, attention='cbam', reduction=16, spatial_kernel=(1,7)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel), stride=(1,stride),
                               padding=(0, kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,kernel), stride=1,
                               padding=(0, kernel//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # attach attention
        self.attn = None
        if attention == 'se':
            self.attn = SEBlock(out_channels, reduction=reduction)
        elif attention == 'cbam':
            self.attn = CBAM(out_channels, reduction=reduction, spatial_kernel=spatial_kernel)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.attn is not None:
            out = self.attn(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ECGSMARTNET_Attention(nn.Module):
    def __init__(self, num_classes=2, kernel=7, kernel1=3, num_leads=12, dropout=False,
                 attention='se', reduction=16, spatial_kernel=(1,7)):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,kernel), stride=(1,2), padding=(0,kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        # layer1: low-level morphology (optionally no attention)
        self.layer1 = self.make_layer(64,  2, stride=1, kernel=kernel1,
                                      attention=None, reduction=reduction, spatial_kernel=spatial_kernel)
        # layers 2-4: attach attention
        self.layer2 = self.make_layer(128, 2, stride=2, kernel=kernel1,
                                      attention=attention, reduction=reduction, spatial_kernel=spatial_kernel)
        self.layer3 = self.make_layer(256, 2, stride=2, kernel=kernel1,
                                      attention=attention, reduction=reduction, spatial_kernel=spatial_kernel)
        self.layer4 = self.make_layer(512, 2, stride=2, kernel=kernel1,
                                      attention=attention, reduction=reduction, spatial_kernel=spatial_kernel)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(num_leads,1), stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        # Optional: one more attention after cross-lead fusion
        self.post_lead_attn = None
        if attention is not None:
            if attention == 'se':
                self.post_lead_attn = SEBlock(512, reduction=reduction)
            elif attention == 'cbam':
                # Here, spatial (1,7) is purely temporal; after conv2, H=1 (lead collapsed), so (1,7) is perfect
                self.post_lead_attn = CBAM(512, reduction=reduction, spatial_kernel=(1,7))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = dropout
        self.do = nn.Dropout(p=0.5)

    def make_layer(self, out_channels, num_blocks, stride, kernel, attention, reduction, spatial_kernel):
        layers = []
        layers.append(ResidualBlock2D_Attention(self.in_channels, out_channels, stride, kernel,
                                      attention=attention, reduction=reduction, spatial_kernel=spatial_kernel))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D_Attention(self.in_channels, out_channels, 1, kernel,
                                          attention=attention, reduction=reduction, spatial_kernel=spatial_kernel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.relu(self.bn2(self.conv2(out)))  # cross-lead fusion

        if self.post_lead_attn is not None:
            out = self.post_lead_attn(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.dropout:
            out = self.do(out)
        return out
  
########################################################################################

# Temporal ResNet-18 Architecture ######################################################
class Temporal(nn.Module):
    def __init__(self, num_classes=2):
        super(Temporal, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,7), stride=(1,2), padding=(0,3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []

        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
########################################################################################

# Temporal ResNet-34 Architecture ######################################################
class TemporalResNet34(nn.Module):
    def __init__(self, num_classes=2):
        super(TemporalResNet34, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,7), stride=(1,2), padding=(0,3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        # ResNet-34 uses [3, 4, 6, 3] blocks instead of [2, 2, 2, 2]
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.layer2 = self.make_layer(128, 4, stride=2)
        self.layer3 = self.make_layer(256, 6, stride=2)
        self.layer4 = self.make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride):
        layers = []

        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
########################################################################################

# Pretrained ResNet-18 Architecture ####################################################
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
########################################################################################

# ResNet-34 Architecture (From Scratch) ###############################################
class ResNet34(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights=None)  # Train from scratch, no pretrained weights
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
########################################################################################