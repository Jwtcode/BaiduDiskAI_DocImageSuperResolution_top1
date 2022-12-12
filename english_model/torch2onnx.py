from torch import nn
import torch.onnx
from model.rcan import RCAN
from option import args
from pdb import set_trace as stx
import os

      
def test():
  
    # create model
    model =RCAN(args)
    model_dict=model.state_dict()
    #37.03 ->36.03
#     model.cuda()
#36.81,36.80,37.03,37.033,37.007,37.015
    pthfile = './checkpoint/model_epoch_41_psnr_37.03.pth'
    loaded_model = torch.load(pthfile, map_location=torch.device('cpu'))
    model.load_state_dict(loaded_model['state_dict'])
    model.eval()
    #x = torch.randn(1, 3, 1024 ,1024,device="cuda:0")
    # x = torch.randn(1, 3, 1024 ,1024,device="cpu")
    x = torch.randn(1, 3,512,512)
    input_name = 'input'
    output_name1 = 'output1'
    output_name2 = 'output2'
    torch.onnx.export(model,               # model being run
                  x,                         # model input 
                  "./e.onnx", 
                  verbose=True,# where to save the model (can be a file or file-like object)                  
                  opset_version=11,          # the ONNX version to export the model to                  
                  input_names = [input_name],   # the model's input names
                  output_names = [output_name1,output_name2], # the model's output names
                  dynamic_axes= {
                        input_name: {0: 'batch_size',2:'in_width',3:'int_height'},
                        output_name1: {0: 'batch_size',2:'in_width',3:'int_height'},
                        output_name2: {0: 'batch_size',2:'in_width',3:'int_height'}
                       }
                  )
if __name__ == "__main__":

    test()