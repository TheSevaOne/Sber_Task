import torch.nn.functional as F
import torchvision.transforms.functional as Transforms
import torch.nn as nn
import torch
class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Layer, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
    def forward(self, x):
        return self.conv(x)


class SegmenterModel(nn.Module):
    def __init__(self,features=[64, 128, 256, 512],in_channels=3):
        super(SegmenterModel, self).__init__()
        self.upscale = nn.ModuleList()
        self.downscale = nn.ModuleList()

        for elem in features: 
            self.downscale.append(Layer(in_channels, elem))
            in_channels = elem
        for elem  in reversed(features):
            self.upscale.append(nn.ConvTranspose2d(
                    elem*2, elem, kernel_size=2, stride=2))
            self.upscale.append(
                Layer(elem*2,elem) 
            )
        self.pooling = nn.MaxPool2d(2, 2)
        self.bottleneck = Layer(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],4 , 1)

    def forward(self, x):	      
        stack= []
        for down in self.downscale: 
            x = down(x)
            stack.append(x)
            x = self.pooling(x) 

        x = self.bottleneck(x) 

        stack = stack[::-1]
        for idx in range(0, len(self.upscale), 2):
            x = self.upscale[idx](x)
            stack_ = stack[idx//2]
            if x.shape != stack_.shape:
                x = Transforms.resize(x, size=stack_.shape[2:])
            concat_skip = torch.cat((stack_, x), dim=1)
            x = self.upscale[idx+1](concat_skip)
        return self.final_conv(x)
    
    def predict(self, x):
    
        y = self.forward(x.unsqueeze(0).cuda())
        return (y > 0).squeeze(0).squeeze(0).float().cuda()



def inference_forward(image,model,targets=None):
        if targets is None:
            targets_shape = list(image.shape) 
            targets_shape[1] = 4
            targets = torch.zeros(*targets_shape)
            image = image.to('cuda:0')
            outputs = model(image)
            loss = None
            outputs= torch.sigmoid( outputs )
            outputs = outputs.detach().cpu().numpy()
        
        return loss, outputs