import os
import cv2
import datetime
import time
import argparse
import transforms as T
import torch

from visualization_utils.config import METADATA_STUFF_CLASSES, METADATA_STUFF_COLORS
from visualization_utils.visualizer import Visualizer
from visualization_utils.color_mode import ColorMode
from datasets.mapillary import MapillarySegmentation

IMAGE_DIR = '/home/adeshkin/projects/tools/Mapillary/2019-01-31-15-53-26_kia_velo_gps_time_stereo_left'
MASK_DIR = '/home/adeshkin/projects/semantic_segmentation/pytorch-segmentation/vis/vis_nkbvs_resnet34_pyramid_oc_1920_1080_layer3_pretrained_backbone_best'
SAVE_DIR = './amazing_nkbvs_1920_1080'
MODEL_PATH = '/home/adeshkin/projects/semantic_segmentation/pytorch-segmentation/pretrained_models/mapillary_resnet34_pyramid_oc_1920_1080_layer3_pretrained_backbone/model_best.pth'
NUM_CLASSES = 66

def parse_args():
    parser = argparse.ArgumentParser(description='Visualization of semantic segmentation masks with class names')
    
    parser.add_argument('--image-dir', metavar='DIR', default=IMAGE_DIR, help='path to images')
    parser.add_argument('--mask-dir', metavar='DIR', default=MASK_DIR, help='path to masks')
    parser.add_argument('--save-dir', default=SAVE_DIR, help='path where to amazing outputs')
    parser.add_argument('--image-set', default='test', help='train, val or test')
    parser.add_argument('--arch', metavar='ARCH', default='resnet34_pyramid_oc_layer3', help='model architecture')
    parser.add_argument('--model-path', default=MODEL_PATH, help='model path')
    parser.add_argument('--device', default='cuda:0', help='device')
    
    args = parser.parse_args()
    
    return args

def vis_one_image(image, mask, save_name, save_dir):  
    # Convert image from OpenCV BGR format to Matplotlib RGB format.
    #image = image[:, :, ::-1]
        
    visualizer = Visualizer(image, METADATA_STUFF_CLASSES, METADATA_STUFF_COLORS, instance_mode=ColorMode.SEGMENTATION)
    vis_output = visualizer.draw_sem_seg(mask)
    
    save_path = os.path.join(save_dir, save_name)
    vis_output.save(save_path)

    
def vis_from_directory(image_dir, mask_dir, save_dir, image_paths, mask_paths):
    for i, (image_name, mask_name) in enumerate(zip(image_paths, mask_paths)):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        vis_one_image(image, mask, image_name, save_dir)
        
        if i % 10 == 9:
            print(f'{i} / {len(image_paths)} : {image_name}')
            
def vis_from_model(model, data_loader, device, save_dir):            
    model.eval()
    model.to(device)
    with torch.no_grad(): 
        for i, (image, _) in enumerate(data_loader):
            image_cuda = image.to(device)
            
            output = model(image_cuda)
            output = output['out'].argmax(1)
            
            image = image[0].permute(1,2,0)
            mask = output[0].cpu()
            
            if i < 10:
                image_name = f'000{i}.png'
            elif i < 100:
                image_name = f'00{i}.png'
            elif i < 1000:
                image_name = f'0{i}.png'
            elif i < 10000:
                image_name = f'{i}.png'
                
            vis_one_image(image, mask, image_name, save_dir)
        
            if i % 10 == 9:
                print(f'{i} / {len(data_loader)} : {image_name}')
                
def get_paths(file_dir):
    paths = os.listdir(file_dir)
    paths.sort()
    
    return paths

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    image_paths = get_paths(args.image_dir)
    print(f"Visualization:\n image_dir: {args.image_dir}")
    start_time = time.time()
    
    if args.model_path:
        if args.arch == 'resnet34_base_oc_layer3':
            from network.model import get_resnet34_base_oc_layer3 as get_model
        elif args.arch == 'resnet34_pyramid_oc_layer3':
            from network.model import get_resnet34_pyramid_oc_layer3 as get_model
        elif args.arch == 'resnet34_asp_oc_layer3':
            from network.model import get_resnet34_asp_oc_layer3 as get_model
        
        model = get_model(NUM_CLASSES)
        model_name = get_model.__name__[4:]

        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        dataset_test = MapillarySegmentation(image_set=args.image_set, transforms=get_transform())
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1)
        
        print(f"=> model: {model_name}")
        print(len(data_loader_test))
        vis_from_model(model, data_loader_test, args.device, args.save_dir)
    else:
        print(f" mask_dir: {args.mask_dir}")
        mask_paths = get_paths(args.mask_dir)
        vis_from_directory(args.image_dir, args.mask_dir, args.save_dir, image_paths, mask_paths)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time))) 
    print('Visualization time {}'.format(total_time_str)) 
    print(f'save_dir: {args.save_dir}')
    
if __name__ == "__main__":
    args = parse_args()
    main(args)