import torch
from torch.autograd import Variable
from crnet import *
from number_utils import *
import numpy as np
import cv2
import argparse
import pickle as pkl
import random
import os

def arg_parse():
    """
    Parse arguments to the detect module
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", dest="images", help="Image file or image directory to peform detction on",
                        default="9japlate1.jpg", type=str)
    parser.add_argument("--dest", dest="dest", help="Destination to store detections", default="dest", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object confidence used to filter predictions",
                        default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    
    return parser.parse_args()

def load_classes(path):
    classes = [i.strip() for i in open(path, 'r').readlines() if i!="\n"]
    return classes


def image_detection():
    # Detection Phase
    try:
        imlist = [os.path.join(os.path.realpath('.'), images, img) for img in os.listdir(images)]
    except NotADirectoryError:
        imlist = [os.path.join(os.path.realpath('.'), images)]
    except FileNotFoundError:
        print("No file or directory with name {}".format(images))
        exit()

    if not os.path.exists(args.dest):
        os.mkdir(args.dest)

    loaded_ims = [cv2.imread(x) for x in imlist]

    #Pytorch Variables for images
    im_batches = []
    for x in range(len(imlist)):
        img = prep_image(loaded_ims[x], inp_dim, inp_dim2)
        im_batches.append(img)

    #List contianing dimensions of original images
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:
        img_dim_list = im_dim_list.cuda()

    leftover = 0
    if (len(im_dim_list)%batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(mlist)//batch_size + leftover
        im_batches = [torch.cat((im_batches[i*batch_size:min((i+1)*batch_size, len(im_batches))]))
                      for i in range(num_batches)]


    write = 0
    for i, batch in enumerate(im_batches):
        #load the image
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
            #print(f"Before NMS: {prediction.shape}")

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)
        #print(f"After NMS: {prediction.shape}")
        if type(prediction)==int: # If no detection
            print("No detection")
            #for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
            #    im_id = i*batch_size + im_num
            continue

        prediction[:,0] += i*batch_size

        if not write:
            output = prediction
            write = 1

        else:
            output = torch.cat((output, prediction))

        #for im_num, image in enumerate(imlist[i*batch_size: min((i+1)*batch_size, len(imlist))]):
        #    im_id = i*batch_size + im_num
        #    objs = [classes[int(x[-1])] for x in output if int(x[0])==im_id]
        #if CUDA:
        #    torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made.")
        exit()

    # Rescale output to match the dimension of the original image
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor_h = (inp_dim/im_dim_list[:,1]).view(-1,1)
    scaling_factor_w = (inp_dim2/im_dim_list[:,0]).view(-1,1)
    output[:, [1,3]] -= (inp_dim2 - scaling_factor_w*im_dim_list[:,0].view(-1,1))/2
    output[:, [2,4]] -= (inp_dim - scaling_factor_h*im_dim_list[:,1].view(-1,1))/2
    
    output[:, [1,3]] /= scaling_factor_w
    output[:, [2,4]] /= scaling_factor_h

    # clip bounding box for those that extend outside the detected image
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i, 0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i, 1])
    
    number = sorted(
            [(classes[int(x[-1])], int(x[1])) for x in output],
            key = lambda x: x[1]
        )
        
    if os.path.exists("./dest/license_plate.txt"):
        with open("./dest/license_plate.txt", 'a') as f:
            f.write("".join([str(i[0]) for i in number])+"\n\n")
    else:
        with open("./dest/license_plate.txt", 'a') as f:
            f.write("".join([str(i[0]) for i in number])+"\n\n")
            
    print("".join([str(i[0]) for i in number]))
    result = list(map(lambda x: overlay(x, loaded_ims), output))
    
    det_names = list(map(out_path, imlist))
    list(map(cv2.imwrite, det_names, result))

def overlay(x, result):
    """
    Overlays the image with bounding box and image class.
    """
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = result[int(x[0])] if isinstance(result, list) else result
    color = [0, 0, 255] #random.choice(colors)
    cls_ = int(x[-1])
    label = "{0}".format(classes[cls_])
    #print((c1,c2), label)
    img = cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0]+3, c1[1]+t_size[1]+4
    img = cv2.rectangle(img, c1, c2, [255, 0, 0], -1)
    img = cv2.putText(img, label, (c1[0], c1[1]+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)
    return img


def out_path(x):
    return f"./dest/det_{os.path.basename(x)}"
    #x = os.path.normpath(x)
    #temp = x.split(os.path.sep)
    #s = f"{os.path.sep}".join(temp[:-1])
    #return f"{os.path.sep}".join([s, args.dest, "det_"+os.path.basename(x)])

if __name__ == "__main__":
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thresh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 35
    classes = load_classes("./lp-recognition.names")
    #colors = pkl.load(open("pallete", 'rb'))

    print("Loading network...")
    path = r"./crnet.pt"
    model = torch.load(path)
    print("Network loaded successfully")

    inp_dim = int(model.net_info["height"])
    inp_dim2 = int(model.net_info["width"])
    assert (inp_dim%32 == 0) and (inp_dim > 32)
    
    model.eval()
    image_detection()
    torch.cuda.empty_cache()
