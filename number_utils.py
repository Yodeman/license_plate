import torch
import torch.nn as nn
import numpy as np
import cv2

def parse_cfg(cfgpath):
    """
    cfgFile: path to CR-NET configuration file.

    Returns: A list of blocks. Each block describes a block on the neural
             network to bebuilt. Block is represented as list of dictionaries.
             
    """

    block = {}
    blocks = []

    with open(cfgpath, 'r') as f:
        lines = f.readlines()
        lines = [x.strip(" \n") for x in lines if len(x) and x[0] != '#']

        for line in lines:
            if line.startswith('['):
                if len(block):
                    blocks.append(block)
                    block = {}
                block['type'] = line.strip(" []")
            elif line:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)
    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3

    for idx, block in enumerate(blocks[1:]):
        if (block["type"] == "convolutional"):
            module = nn.Sequential()
            activation = block["activation"]
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except KeyError:
                batch_normalize = 0
                bias = True

            filters = int(block["filters"])
            padding = int(block["pad"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])

            if padding:
                pad = (kernel_size - 1)//2
            else:
                pad = 0

            conv = nn.Conv2d(
                    in_channels=prev_filters, out_channels=filters,
                    kernel_size=kernel_size, stride=stride, padding=pad,
                    bias=bias
                )
            module.add_module(f"conv{idx}", conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters) #, track_running_stats=False
                module.add_module(f"batch_norm{idx}", bn)

            if (activation == "leaky"):
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f"leaky{idx}", act)

            module_list.append(module)
            prev_filters = filters
            
        elif (block["type"] == "maxpool"):
            pool_size = int(block["size"])
            stride = int(block["stride"])

            max_pool = nn.MaxPool2d(pool_size, stride)
            module_list.append(max_pool)

        elif (block["type"] == "region"):
            anchors = list(map(float, block["anchors"].split(',')))
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            num_classes = int(block["classes"])
            object_scale = int(block["object_scale"])
            noobject_scale = int(block["noobject_scale"])
            class_scale = int(block["class_scale"])
            coord_scale = int(block["coord_scale"])
            thresh = float(block["thresh"])

            detection = DetectionLayer(
                    anchors, num_classes, object_scale, noobject_scale,
                    class_scale, coord_scale, thresh
                )
            module_list.append(detection)

    return (net_info, module_list)

class DetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes, object_scale, noobject_scale,
                 class_scale, coord_scale, thresh):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.object_scale = object_scale
        self.noobject_scale = noobject_scale
        self.class_scale = class_scale
        self.coord_scale = coord_scale
        self.thresh = thresh


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    """
    Transform prediction from B x 200 x 16 x 44 into B x num_bboxes x box_attr.
    B - batch size
    num_bboxes - number of bounding boxes
    box_attr - bounding box attributes
               (center_x, center_y, h, w, objectness_score, class probabilites)
    """
    batch_size = prediction.shape[0]
    h = prediction.shape[2]
    w = prediction.shape[3]
    stride = inp_dim//h
    num_anchors = len(anchors)
    box_attr = 5+num_classes
    
    prediction = prediction.view(batch_size, box_attr*num_anchors, h*w)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, h*w*num_anchors, box_attr)
    
    #anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    #Perform sigmoid on center_x, center_y, and objectness confidence
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    #print(torch.max(prediction[:, :, 4], 1))
    
    #Add the center of offsets
    grid_x = np.arange(w)
    grid_h = np.arange(h)
    a, b = np.meshgrid(grid_x, grid_h)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset
    
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
        
    anchors = anchors.repeat(w*h, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5:5+num_classes] = torch.softmax(prediction[:,:,5:5+num_classes], 2)
    
    prediction[:,:,:4] *= stride
    
    return prediction
    
def bboxes_iou(box1, box2):
    """
    Returns the IoU of a box with other boxes.
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get coordinates of the intersecting rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    #Intersection Area
    inter_area = torch.clamp(inter_rect_x2-inter_rect_x1+1, min=0) * torch.clamp(inter_rect_y2-inter_rect_y1+1, min=0)
    
    #Union Area
    b1_area = (b1_x2 - b1_x1+1)*(b1_y2 - b1_y1+1)
    b2_area = (b2_x2 - b2_x1+1)*(b2_y2 - b2_y1+1)
    
    iou = inter_area/(b1_area+b2_area-inter_area)
    
    return iou
            
def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    #result = tensor.new(unique_tensor.shape)
    #result.copy_(unique_tensor)
    
    return unique_tensor #result 

def letterbox_image(img, inp_dim):
    """ Resize image with unchanged aspect ratio using padding."""
    img_h, img_w = img.shape[0], img.shape[1]
    h, w = inp_dim
    new_w = int(img_w * (w/img_w)) #min(w/img_w, h/img_h)
    new_h = int(img_h * (h/img_h)) #min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((h, w, 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h, (w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim1, inp_dim2):
    """
    Prepare image to match the expectation of the network.
    """
    #img = cv2.resize(img, (inp_dim, inp_dim))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose((2,0,1))
    #img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    img = (letterbox_image(img, (inp_dim1, inp_dim2)))
    img = img[:,:,::-1].transpose((2,0,1)).copy() #img[:, :, ::-1] --> BGR2RGB
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    #print(img.shape)
    return img
    
def write_results(prediction, confidence, num_classes, nms_conf=0.4):

    """
    This function applies non-maximum supppression on the prediction
    and returns a tensor of batch_size x bbox_attr.
    The box attributes is a row tensor that contains the bounding box
    coordinates, probability
    
    Parameters:
        prediction - a tensor of batch_size x num_bboxes x box_attr
        confidence - probability threshold of a box containing the
                     expected object.
        num_classes - number of objects.
        nms_conf    - non-max supppression threshold.
        
    
    """
    
    # set all prediction less than confidence to zero
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    # transform bounding box coordinates to diagonal corners
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)    
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)    
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)    
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)
    write = False
    
    for ind in range(batch_size):
        image_pred = prediction[ind]
        # class with highest score
        max_conf_score, max_conf_ind = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        max_conf_ind = max_conf_ind.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf_score, max_conf_ind)
        image_pred = torch.cat(seq, 1)
        
        # get rid of boxes without object
        non_zero_ind = torch.nonzero(image_pred[:,4])
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        #print(image_pred_.shape)
        if not image_pred_.shape[0]:
            continue
        #Get various classes detected in the image
        img_classes = unique(image_pred_[:,-1]) # -1 contains the class index
        
        for cls_ in img_classes:
            # Perform NMS
            
            #get detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1]==cls_).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            # sort the detections such that the entry with maximum objectness
            # confidence is at the top
            _, conf_sort_index = torch.sort(image_pred_class[:,4], descending=True)
            image_pred_class = image_pred_class[conf_sort_index]
            
            idx = image_pred_class.size(0) #number of detections
            
            for i in range(idx):
                try:
                    ious = bboxes_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except (ValueError,IndexError):
                    break
                    
                # zero out all detections with IoU>threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask
                
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
                
    try:
        return output
    except:
        return 0
