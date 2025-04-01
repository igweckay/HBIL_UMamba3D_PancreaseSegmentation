import nibabel as nib


# This will not be used in the final code, but is here for reference
def _load_nifti_image(file_path):
    """
    Load a NIfTI image from a file path.
    
    Args:
        file_path (str): Path to the NIfTI file.
    
    Returns:
        tuple: A tuple containing the image data (numpy array) and the affine transformation matrix.
    """
    scan = nib.load(file_path)
    img = scan.get_fdata().astype('float32')
    return img, scan.affine

def _save_nifti_image(image, affine, file_path): 
    """
    Save a NIfTI image to a file path.
    
    Args:
        image (numpy array): The image data to save.
        affine (numpy array): The affine transformation matrix.
        file_path (str): Path to save the NIfTI file. Must end with '.nii' or '.nii.gz'.
    """
    assert file_path.endswith('.nii') or file_path.endswith('.nii.gz'), "File path must end with '.nii' or '.nii.gz'"
    scan = nib.Nifti1Image(image, affine)
    nib.save(scan, file_path)

def get_device(gpu_device_index=0):
    """
    Get the device to run the model on (GPU or CPU).
    
    Returns:
        torch.device: The device object.
    """
    import os, torch
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device('cuda:%d'%gpu_device_index if torch.cuda.is_available() else 'cpu')
    return device
    
def get_model(checkpoint_path='./UMamba3D.ckpt'):
    """
    Craete UMamba3D and load a pre-trained model from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
        torch.nn.Module: The loaded model.
    """
    from collections import OrderedDict
    import torch
    from torch import nn
    from umamba import UMambaBot  # Assuming UNet3D is defined in models.py

    model = UMambaBot(
        input_channels=1,  # Number of input channels (e.g., grayscale volumes)
        n_stages=4,  # Number of encoder/decoder stages
        features_per_stage=[16, 32, 64, 128],  # Features at each stage
        conv_op=nn.Conv3d,  # 3D convolutions
        kernel_sizes=[(3, 3, 3)] * 4,  # Kernel sizes per stage
        strides=[(1, 1, 1)] + [(2, 2, 2)] * 3,  # Strides for downsampling
        n_conv_per_stage=2,  # Number of convolutions per stage
        num_classes=1,  # Number of output segmentation classes
        n_conv_per_stage_decoder=2,  # Convolutions per decoder stage
        conv_bias=True,
        norm_op=nn.BatchNorm3d,  # Batch normalization for 3D
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,  # Activation function
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,  # Use final output only
    )
    
    stat_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['state_dict']
    new_stat_dict = OrderedDict()
    for k, v in stat_dict.items():
        new_stat_dict[k.replace('core_model.', '')] = v
        
    model.load_state_dict(new_stat_dict)
    model.eval()
    
    return model


def get_pre_post_transforms():
    import numpy as np
    from monai.transforms import (
        AsDiscrete,
        Compose,
        Resize,
        ScaleIntensityRange,
        GridPatch,
        ToTensor,
        Activations,
        LoadImage,
    )
    # Must be modified to match the model input and training parameters
    slice_shape = (256, 256)
    number_of_slices = 32
    label_threshold = 0.5
    
    preprocess_transform = Compose([
        LoadImage(dtype=np.float32, ensure_channel_first=True),
        ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        Resize(spatial_size=(*slice_shape, -1), mode='bilinear'),
        GridPatch(patch_size=(0,0, number_of_slices), overlap=0, pad_mode='reflect')
        ])

    postprocess_transform = Compose([ToTensor(), 
                                    Activations(sigmoid=True), 
                                    AsDiscrete(threshold=label_threshold)])
    
    return preprocess_transform, postprocess_transform
    

def save_prediction(pred, inp, post_trans, output_path):
    original_affine = inp.meta['original_affine']
    originlal_shape = tuple(inp.meta['dim'][1:4])
    
    from einops import rearrange
    from monai.transforms import Resize
    resize_transform = Resize(spatial_size=(*originlal_shape[:2], -1), mode='nearest')
    pred = rearrange(pred, 'b c h w d -> c h w (b d)')
    pred = pred[...,:originlal_shape[-1]]
    pred = resize_transform(pred)
    pred = post_trans(pred).squeeze().cpu().numpy()
    
    _save_nifti_image(pred, affine=original_affine, file_path=output_path)