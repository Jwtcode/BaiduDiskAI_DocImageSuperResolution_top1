import torch
from torch import nn
import torch.onnx
from resnet18 import VGG16
from option import args
from pdb import set_trace as stx
import os

      
def test():
  
    # create model
    model =VGG16()
    model_dict=model.state_dict()
    model.cuda()
    pthfile = './checkpoint/model_best.pth'
    loaded_model = torch.load(pthfile)
    model.load_state_dict(loaded_model['state_dict'])
    model.eval()
    #x = torch.randn(1, 3, 1024 ,1024,device="cuda:0")
    # x = torch.randn(1, 3, 1024 ,1024,device="cpu")
    x = (torch.randn(1,3,16,16, device='cuda'))
    input_name = 'input'
    output_name = 'output'

    torch.onnx.export(model,               # model being run
                  x,                         # model input 
                  "./vgg.onnx", 
                  verbose=True,# where to save the model (can be a file or file-like object)                  
                  opset_version=11,          # the ONNX version to export the model to                  
                  input_names = [input_name],   # the model's input names
                  output_names = [output_name], # the model's output names
                  )
if __name__ == "__main__":

    test()

