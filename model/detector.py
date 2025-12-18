import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_detector_model(num_classes, weights_path=None, device=None):
    '''
    Builds the Faster R-CNN model for object detection.
    
    Args:
        num_classes (int): Number of classes including background.
        weights_path (str, optional): Path to .pth file to load trained weights.
        device (torch.device, optional): Device to load weights onto.
        
    Returns:
        model (torch.nn.Module): The detector model.
    '''
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if weights_path:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f'Loading detector weights from {weights_path}...')
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    
    return model