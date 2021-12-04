import numpy as np
import torch
import torch.nn as nn
from number_utils import *

class CRNET(nn.Module):
    
    """
    The CR-NET architecture as proposed in the research paper
    by Rayson Laroca et. al
    """

    def __init__(self, cfgpath):
        super(CRNET, self).__init__()
        self.blocks = parse_cfg(cfgpath)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):

        modules = self.blocks[1:]

        for idx, module in enumerate(modules):
            module_type = module["type"]
            
            if module_type in ("convolutional", "maxpool"):
                x = self.module_list[idx](x)

            elif (module_type == "region"):
                detection_layer = self.module_list[idx]
                anchors = detection_layer.anchors
                inp_dim = int(self.net_info["height"])
                num_classes = detection_layer.num_classes

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
        return x

    def load_weights(self, weightpath):

        fp = open(weightpath, "rb")
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0 #pointer to starting and ending point of parameters

        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    #Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy data into model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Copy convolutional layer bias
                    num_conv_bias = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_conv_bias])
                    ptr += num_conv_bias

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                # Copy convolutional layer weights
                num_conv_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_conv_weights])
                ptr += num_conv_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                    

##def get_test_input():
##    img = cv.imread("./9japlate1.jpg", cv.IMREAD_COLOR)
##    x = img[:,:,::-1].transpose((2,0,1))
##    x = x[np.newaxis,:,:,:]/255.0
##    x = torch.from_numpy(x).float()
##    return x

if __name__ == "__main__":
##    
##    from pprint import pprint
##
    #import cv2 as cv
    #from torch.autograd import Variable
      
    cfgpath = "./lp-recognition.cfg"
    weightpath = "./lp-recognition.weights"
    #x = torch.randn((3, 128, 352))
    #x = get_test_input()
    
##    print(f"Input Shape: {x.shape}")
    model = CRNET(cfgpath)
##    print(model)
    model.load_weights(weightpath)

    #x = prep_image(x, 128, 352)
    #model.eval()
    
    #output = model(Variable(x), False)
    #o = write_results(output, 0.92, 35)
    #print(f"Output Shape:{output.shape}")
    #print(o.shape)
    #print(torch.max(output[:,:,4]))
    #blocks = parse_cfg(cfgpath)
    #pprint(blocks)
    torch.save(model, "./crnet.pt")
