from torch import nn
from .mobilenetv3 import MobileNetV3LargeEncoder, MobileNetV3LargeEncoder_CBAM

class FaceEncoder(nn.Module):
    def __init__(self, pretrained: bool = False,in_ch: int = 3):
        super().__init__()
        self.face_encoder = MobileNetV3LargeEncoder(pretrained=pretrained,in_ch=in_ch)
        
    def forward(self, input):
        fc0, fc1, fc2, fc3 = self.face_encoder(input)
        return fc0, fc1, fc2, fc3
    
class FaceEncoder_CBAM(nn.Module):
    def __init__(self, pretrained: bool = False,in_ch: int = 3):
        super().__init__()
        self.face_encoder = MobileNetV3LargeEncoder_CBAM(pretrained=pretrained,in_ch=in_ch)
        
    def forward(self, input):
        fc0, fc1, fc2, fc3 = self.face_encoder(input)
        return fc0, fc1, fc2, fc3