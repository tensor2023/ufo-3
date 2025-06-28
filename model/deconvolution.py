import torch
import math
import torch.nn as nn
# from .unet import GraphCNNUnet
#from .unet import Block2

from .interpolation import Interpolation
from monai.networks.nets import UNet
import torch.nn.functional as F
from abc import ABC, abstractmethod
import os


class Deconvolution(torch.nn.Module):
    def __init__(self,  filter_start, kernel_sizeSph, kernel_sizeSpa, n_fodf_coff, n_extra_trapped, normalize, conv_name, isoSpa, feature_in=1):
        """Separate fodf_coffvariant and extra_trappedriant features from the deconvolved model
        Args:
            x (:obj:`torch.Tensor`): input. 
            shellSampling (:obj:`sampling.ShellSampling`): Input sampling scheme
            graphSampling (:obj:`sampling.Sampling`): Interpolation grid scheme
            filter_start (int): First intermediate channel (then multiply by 2)
            kernel_sizeSph (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            kernel_sizeSpa (int): Size of the spatial kernel
            n_fodf_coff (int): Number of fodf_coffvariant deconvolved channel
            n_extra_trapped (int): Number of extra_trappedriant deconvolved channel
            normalize (bool): Normalize the output such that the sum of the SHC of order and degree 0 of the deconvolved channels is math.sqrt(4 * math.pi)
        """
        super(Deconvolution, self).__init__()
       
        self.n_fodf_coff = n_fodf_coff
        self.n_extra_trapped = 2#n_extra_trapped
        self.normalize = normalize
        
        self.interpolate = Interpolation(conv_name)
        self.eps = 1e-16
        
        self.deconvolve=UNet3D(3072,48,32)#16:156 8:48
        
    def separate(self, x):
        if self.n_extra_trapped != 0:
            x_3 = F.softplus(x[:, 44:-1])
        else:
            x_3 = None
        # to_norm = self.eps + x_3[:, 0:1]* math.sqrt(4 * math.pi)+ torch.sum(x_3[:, 1:], axis=1, keepdim=True)#.clamp(0,
        
        # x_3 =x_3 / to_norm
        x_fodf_coff=torch.cat((x_3[:,0:1],x[:,:44]),dim=1)
        x_extra_trapped=x_3[:,1:]
        iso=F.softplus(x[:, -1:]).clamp(1.0e-16,4)

        return x_fodf_coff[:,None], x_extra_trapped,iso
    def norm(self, x_fodf_coff, x_extra_trapped):
        """Separate fodf_coffvariant and extra_trappedriant features from the deconvolved model
        Args:
            x_fodf_coff (:obj:`torch.Tensor`): shc fodf_coffvariant part of the deconvolution 
            x_extra_trapped (:obj:`torch.Tensor`): shc extra_trappedriant part of the deconvolution 
        Returns:
            x_fodf_coff (:obj:`torch.Tensor`): normed shc fodf_coffvariant part of the deconvolution 
            x_extra_trapped (:obj:`torch.Tensor`): normed shc extra_trappedriant part of the deconvolution 
        """
        to_norm = 0
        if self.n_fodf_coff != 0:
            to_norm = to_norm + torch.sum(x_fodf_coff[:, :, 0:1], axis=1, keepdim=True)
        if self.n_extra_trapped != 0:
            to_norm = to_norm* math.sqrt(4 * math.pi) + torch.sum(x_extra_trapped, axis=1, keepdim=True)
        to_norm = to_norm  + self.eps
        if self.n_fodf_coff != 0:
            x_fodf_coff = x_fodf_coff / to_norm
        if self.n_extra_trapped != 0:
            x_extra_trapped = x_extra_trapped / to_norm
        return x_fodf_coff, x_extra_trapped


    def forward(self, x,sampling2sampling):
        x = self.interpolate(x,sampling2sampling) 
      
        x_deconvolved = self.deconvolve(x.squeeze(1))
        x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped,iso = self.separate(x_deconvolved) 
        if self.n_extra_trapped != 0:
            x_deconvolved_extra_trapped_shc = (x_deconvolved_extra_trapped* math.sqrt(4 * math.pi))[:, :, None] 
        else:
            x_deconvolved_extra_trapped_shc = None#
        
        # Normalize
        if self.normalize:
            x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc = self.norm(x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc)
        return x_deconvolved_fodf_coff_shc, x_deconvolved_extra_trapped_shc,iso

class BaseModel(nn.Module, ABC):
    r"""
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()
        self.best_loss = 1000000

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def test(self):
        """
        To be implemented by the subclass so that
        models can perform a forward propagation
        :return:
        """
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        r"""
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError("No checkpoint file to be restored.")

        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        # Restore model weights
        self.load_state_dict(ckpt_dict['model_state_dict'])

        # Restore optimizer status if existing. Evaluation doesn't need this
        # TODO return optimizer?????
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])

        # Return global step
        return ckpt_dict['epoch']

    def save_checkpoint(self,
                        directory,
                        epoch, loss,
                        optimizer=None,
                        name=None):
        r"""
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            epoch (int): The training. epoch
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        # Create directory to save to
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Build checkpoint dict to save.
        ckpt_dict = {
            'model_state_dict':
                self.state_dict(),
            'optimizer_state_dict':
                optimizer.state_dict() if optimizer is not None else None,
            'epoch':
                epoch
        }

        # Save the file with specific name
        if name is None:
            name = "{}_{}_epoch.pth".format(
                os.path.basename(directory),  # netD or netG
                'last')

        torch.save(ckpt_dict, os.path.join(directory, name))
        if self.best_loss > loss:
            self.best_loss = loss
            name = "{}_BEST.pth".format(
                os.path.basename(directory))
            torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        r"""
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters()
                                   if p.rfodf_coffres_grad)

        return num_total_params, num_trainable_params

    def inference(self, input_tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().detach()

class UNet3D(BaseModel):
    """
    Implementations based on the Unet3D paper: https://arxiv.org/abs/1606.06650
    """

    def __init__(self, in_channels, n_classes, base_n_filter):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.BatchNorm3d(self.base_n_filter)

        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter * 2, self.base_n_filter * 2)
        self.inorm3d_c2 = nn.BatchNorm3d(self.base_n_filter * 2)

        self.conv3d_c3 = nn.Conv3d(self.base_n_filter * 2, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter * 4, self.base_n_filter * 4)
        self.inorm3d_c3 = nn.BatchNorm3d(self.base_n_filter * 4)

        self.conv3d_c4 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter * 8, self.base_n_filter * 8)
        self.inorm3d_c4 = nn.BatchNorm3d(self.base_n_filter * 8)

        self.conv3d_c5 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 16, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter * 16, self.base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu1(self.base_n_filter * 16,
                                                                                             self.base_n_filter * 8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.inorm3d_l0 = nn.BatchNorm3d(self.base_n_filter * 8)

        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter * 16, self.base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter * 16, self.base_n_filter * 8, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 8,
                                                                                             self.base_n_filter * 4)

        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter * 8, self.base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter * 8, self.base_n_filter * 4, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 4,
                                                                                             self.base_n_filter * 2)

        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter * 4, self.base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter * 4, self.base_n_filter * 2, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter * 2,
                                                                                             self.base_n_filter)

        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter * 2, self.base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter * 2, self.n_classes, kernel_size=1, stride=1, padding=0,
                                   bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter * 8, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter * 4, self.n_classes, kernel_size=1, stride=1, padding=0,
                                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=1, stride=1, padding=0, bias=False))
    
    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=1, stride=1, padding=0, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            nn.LeakyReLU(),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(feat_out),
            nn.LeakyReLU())
    def norm_lrelu_upscale_conv_norm_lrelu1(self, feat_in, feat_out):
        return nn.Sequential(
            nn.BatchNorm3d(feat_in),
            nn.LeakyReLU(),
            #nn.Upsample(scale_factor=1, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(feat_out),
            nn.LeakyReLU())
    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        
        out = self.conv3d_c5(out)
        
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
       
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
       
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)
      

        out = self.conv3d_l0(out)
      
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        # #print(out.shape)
        # #print(context_3.shape)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        #ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds2_1x1_conv + ds3_1x1_conv
        #ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum
        seg_layer = out
        return seg_layer

    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        #print("Unet3D test is complete")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias_init=0.001, p_keep_conv=1.0, activation='relu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,stride,padding=1)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, bias_init)
        self.dropout = nn.Dropout3d(1 - p_keep_conv) if p_keep_conv < 1.0 else nn.Identity()
        
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias_init=0.001, p_keep_conv=1.0, activation='relu'):
        super(ConvTransBlock, self).__init__()
        self.kernel_size = 3 # kernel_size
        self.stride = 2 # stride
        
        self.padding = 1 # (self.kernel_size - self.stride + 1) // 2
        
        self.conv_trans = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding=0,
            output_padding=0 
        )
        
        nn.init.kaiming_normal_(self.conv_trans.weight, nonlinearity='relu')
        nn.init.constant_(self.conv_trans.bias, bias_init)
        self.dropout = nn.Dropout3d(1 - p_keep_conv) if p_keep_conv < 1.0 else nn.Identity()
        
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def forward(self, x):
        pad_amount =x.shape[-1]*2
        x = self.conv_trans(x)
        x = x[:,:,:pad_amount,:pad_amount,:pad_amount]
        x = self.activation(x)
        x = self.dropout(x)
        return x
