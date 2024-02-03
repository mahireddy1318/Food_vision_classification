import torch
import torchvision

from torch import nn


def create_effnetb1_model(num_classes:int=101, 
                          seed:int=42):
    """Creates an EfficientNetB1 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 101.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB1 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # Create EffNetB1 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b1(weights=weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    
    return model, transforms
