import torch
import torch.nn as nn
import torchvision.models as models

def get_classifier_model(num_classes, weights_path=None, device=None):
    '''
    Builds a modified ResNet-18 for symbol classification on small images (32x32).
    
    Args:
        num_classes (int): Total unique symbols (e.g., 100+).
        weights_path (str, optional): Path to .pth file to load trained weights.
        device (torch.device, optional): Device to load weights onto.
        
    Returns:
        model (torch.nn.Module): The classifier model.
    '''
    model = models.resnet18(weights='DEFAULT')
    
    # Note: we modify the model for 32x32 inputs (standard ResNet expects 224x224)
    # we change the first conv layer and remove the first maxpool layer to retain more spatial info.
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # we remove the maxpool layer for the same reason 
    model.maxpool = nn.Identity()
    
    # replace the classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if weights_path:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        print(f'Loading classifier weights from {weights_path}...')
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    return model