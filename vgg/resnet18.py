
import torch
import torch.nn as nn
from pdb import set_trace as stx
# class VGG16(nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 8, 3, 1, 1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3, 1, 1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(4, 4)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(8, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(4, 4),
#             nn.Conv2d(16, 16, 3, 1, 1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.fc1 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(16, 1),
#         )
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.fc1(x)
#         x=nn.Sigmoid()(x)
#         return x
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(4,4)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc1(x)
        x=nn.Sigmoid()(x)
        return x
if __name__ == '__main__':
    VGG16 = VGG16()
    input = torch.ones((64, 3, 16, 16))
    output = VGG16(input)
    print(output.shape)