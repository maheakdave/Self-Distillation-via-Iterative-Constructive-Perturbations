from torchvision.models.segmentation import deeplabv3_resnet50,deeplabv3_resnet101,deeplabv3_mobilenet_v3_large
from torchvision.models import ResNet101_Weights,ResNet50_Weights,MobileNet_V3_Large_Weights

def return_deeplabv3(num_classes,backbone = 'resnet50'):
    """
    backbone = {
                    resnet50,
                    resnet101,
                    mobilenet_v3_large
                }
    """
    if backbone == 'resnet50':
        return  deeplabv3_resnet50(weights=None,weights_backbone = ResNet50_Weights.DEFAULT,num_classes = num_classes)
    elif backbone == 'resnet101':
        return deeplabv3_resnet101(weights=None,weights_backbone = ResNet101_Weights.DEFAULT,num_classes = num_classes)
    elif backbone == 'mobile_v3_large':
        return deeplabv3_mobilenet_v3_large(weights=None,weights_backbone = MobileNet_V3_Large_Weights.DEFAULT,num_classes=num_classes)
    else:
        raise Exception('No backbone selected!')
    
