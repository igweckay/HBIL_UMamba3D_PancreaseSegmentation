# %% UMamba3D segmentation
import os
import torch
from utils import get_device, get_model, get_pre_post_transforms, save_prediction
import argparse
parser = argparse.ArgumentParser(description='UMamba3D segmentation')
parser.add_argument('--image_path', '-i', type=str, help='Path to the input NIfTI image', required=True,  metavar='FILE')
parser.add_argument('--output_path', '-o', type=str, help='Path to save the segmented output', required=True,  metavar='FILE')

device, model, pre_trans, post_trans = None, None, None, None
# %%
def prepare():
    """
    Prepare the model and device for segmentation.
    
    """
    global device, model, pre_trans, post_trans
    device = get_device()
    model = get_model()
    model = model.to(device)
    model.eval()
    pre_trans, post_trans = get_pre_post_transforms()

def segment(image_path, output_path):
    """
    Segment a 3D image using the UMamba3D model.
    
    Args:
        image_path (str): Path to the input NIfTI image.
        output_path (str): Path to save the segmented output.
    """
    inp = pre_trans(image_path)
    inp = inp.to(device)
    
    
    with torch.no_grad():
        pred = model(inp)
    
    save_prediction(pred, inp, post_trans, output_path)

# %%
image_path = './TestSample/Image.nii.gz'
output_path = './TestSample/Segment.nii.gz'

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    image_path = args.image_path
    output_path = args.output_path
    
    assert os.path.exists(image_path), f"Image path {image_path} does not exist."
    if os.path.exists(output_path):
        key = input(f"Output path {output_path} already exists. Would you like to overwrite it? (y/n): ")
        if key.lower() != 'y':
            exit(0)
    out_dir = os.path.dirname(output_path)
    if not os.path.isdir(out_dir):
        print(f"Creating directory for output path: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    prepare()
    segment(image_path, output_path)
    print(f"Segmentation completed. Output saved to {output_path}")