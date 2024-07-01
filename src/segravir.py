import torch
import torch.nn as nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp


class ResidualBlock(nn.Module):
    """
    A residual block module that performs residual learning in a neural network.
    
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int): 
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.batchnorm_1 = nn.BatchNorm2d(out_channels)
        self.batchnorm_2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(0.2)

        self.residual_layer = nn.Identity()
    
    def forward(self, feature):
        """
        Forward pass of the residual block.
        
        Args:
            feature (torch.Tensor): The input feature tensor.
                Shape: (Batch_Size, In_Channels, Height, Width)
        
        Returns:
            torch.Tensor: The output feature tensor after passing through the residual block.
                Shape: (Batch_Size, Out_Channels, Height, Width)
        """

        residue = feature

        feature = self.conv_1(feature)
        feature = F.relu(feature)
        feature = self.batchnorm_1(feature)
        feature = self.dropout(feature)

        feature = self.conv_2(feature)
        feature = F.relu(feature)
        feature = self.batchnorm_2(feature)
        feature = self.dropout(feature)

        return feature + self.residual_layer(residue)

class SegRAVIRModel(nn.Module):
    """
    SegRAVIRModel is a convolutional neural network model for semantic segmentation and image reconstruction.

    The model consists of an encoder, a bottleneck layer, and two decoder streams: one for mask prediction and one for image reconstruction.

    Attributes:
        encoder (nn.ModuleList): List of encoder layers.
        bottleneck (nn.ModuleList): List of bottleneck layers.
        decoder_mask (nn.ModuleList): List of decoder layers for mask prediction.
        decoder_reconstruction (nn.ModuleList): List of decoder layers for image reconstruction.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            # (Batch_Size, 3, Height, Width) --> (Batch_Size, 16, Height / 2, Width / 2)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            ResidualBlock(16, 16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),

            # (Batch_Size, 16, Height / 2, Width / 2) --> (Batch_Size, 32, Height / 4, Width / 4)
            ResidualBlock(32, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),

            # (Batch_Size, 32, Height / 4, Width / 4) --> (Batch_Size, 64, Height / 8, Width / 8)
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),

            # (Batch_Size, 64, Height / 8, Width / 8) --> (Batch_Size, 128, Height / 16, Width / 16)
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        ])

        self.bottleneck = nn.ModuleList([
            # (Batch_Size, 128, Height / 16, Width / 16) --> (Batch_Size, 256, Height / 16, Width / 16)
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        ]) 

        self.decoder_mask = nn.ModuleList([
            # (Batch_Size, 256, Height / 16, Width / 16) --> (Batch_Size, 128, Height / 8, Width / 8)
            nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(128, 128),

            # (Batch_Size, 128, Height / 8, Width / 8) --> (Batch_Size, 64, Height / 4, Width / 4)
            nn.ConvTranspose2d(256, 64, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(64, 64),

            # (Batch_Size, 64, Height / 4, Width / 4) --> (Batch_Size, 32, Height / 2, Width / 2)
            nn.ConvTranspose2d(128, 32, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(32, 32),

            # (Batch_Size, 32, Height / 2, Width / 2) --> (Batch_Size, 16, Height, Width)
            nn.ConvTranspose2d(64, 16, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(16, 16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 16, Height, Width)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Softmax(dim=1)
        ])

        self.decoder_reconstruction = nn.ModuleList([
            # (Batch_Size, 256, Height / 16, Width / 16) --> (Batch_Size, 128, Height / 8, Width / 8)
            nn.ConvTranspose2d(256, 128, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(128, 128),

            # (Batch_Size, 128, Height / 8, Width / 8) --> (Batch_Size, 64, Height / 4, Width / 4)
            nn.ConvTranspose2d(256, 64, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(64, 64),

            # (Batch_Size, 64, Height / 4, Width / 4) --> (Batch_Size, 32, Height / 2, Width / 2)
            nn.ConvTranspose2d(128, 32, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(32, 32),

            # (Batch_Size, 32, Height / 2, Width / 2) --> (Batch_Size, 16, Height, Width)
            nn.ConvTranspose2d(64, 16, stride=2, kernel_size=3, padding=1, output_padding=1),
            ResidualBlock(16, 16),

            # (Batch_Size, 16, Height, Width) --> (Batch_Size, 3, Height, Width)
            nn.Conv2d(32, 3, kernel_size=1),
            nn.ReLU()
        ])

    def forward(self, x):
        # x: (Batch_Size, 3, Height, Width)

        skip_encoder = [1, 3, 5, 7]
        skip_connections = []
        i = 0
        for layers in self.encoder:
            x = layers(x)
            if i in skip_encoder:
                skip_connections.append(x)
            i += 1
        
        for layers in self.bottleneck:
            x = layers(x)

        # Copy of the output of the bottleneck
        mask = x
        skip_connections_2 = skip_connections.copy()

        # Main decoder stream 
        skip_decoder = [2, 4, 6, 8]
        i = 0
        for layers in self.decoder_mask:
            if i in skip_decoder:
                mask = torch.cat((mask, skip_connections.pop()), dim=1)
            mask = layers(mask)
            i += 1
        
        # Auxiliary decoder stream
        skip_decoder = [2, 4, 6, 8]
        i = 0
        for layers in self.decoder_reconstruction:
            if i in skip_decoder:
                x = torch.cat((x, skip_connections_2.pop()), dim=1)
            x = layers(x)
            i += 1

        return mask, x

class Loss_Dice_CE_l2(nn.Module):
    """
    A custom loss function that combines Dice coefficient loss, cross-entropy loss, and L2 reconstruction loss.

    Args:
        weight_dice (float): Weight for the Dice coefficient loss. Default is 0.5.
        weight_ce (float): Weight for the cross-entropy loss. Default is 0.3.
        weight_l2 (float): Weight for the L2 reconstruction loss. Default is 0.2.
    """
    def __init__(self, weight_dice=0.5, weight_ce=0.3, weight_l2=0.2):
        super(Loss_Dice_CE_l2, self).__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_l2 = weight_l2

        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            classes=[0, 1, 2]
        )

    def forward(self, logits_mask, pred_img, targets_mask, targets_img):
        """
        Calculates the combined loss using the specified weights.

        Args:
            logits_mask (torch.Tensor): Logits for the mask prediction.
            pred_img (torch.Tensor): Predicted image.
            targets_mask (torch.Tensor): Ground truth mask.
            targets_img (torch.Tensor): Ground truth image.

        Returns:
            torch.Tensor: Combined loss.
        """
        # Dice coefficient loss
        dice_loss = self.dice_loss(logits_mask, targets_mask)

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_mask, targets_mask)

        # L2 reconstruction loss
        l2_loss = F.mse_loss(pred_img, targets_img)

        # Combine the two losses using the specified weight
        combined_loss = self.weight_dice * dice_loss + self.weight_ce * ce_loss + self.weight_l2 * l2_loss

        return combined_loss