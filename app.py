import os
import cv2
import torch
import tempfile
from datetime import datetime
from torch.autograd import Variable
from crnet import *
from plate_utils import *
import streamlit as st


def load_classes(path):
    classes = [i.strip() for i in open(path, 'r').readlines() if i!="\n"]
    return classes

def number_recognition(img):
    
    s_img = prep_image(img, inp_dim, inp_dim2)

    im_dim_list = [(img.shape[1], img.shape[0])]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

    if CUDA:
        s_img = s_img.cuda()
        img_dim_list = im_dim_list.cuda()

    with torch.no_grad():
        prediction = number_model(Variable(s_img), CUDA)

    output = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

    if isinstance(output, int):
        return ""
    
    # Rescale output to match the dimension of the original image
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    scaling_factor_h = (inp_dim/im_dim_list[:,1]).view(-1,1)
    scaling_factor_w = (inp_dim2/im_dim_list[:,0]).view(-1,1)
    output[:, [1,3]] -= (inp_dim2 - scaling_factor_w*im_dim_list[:,0].view(-1,1))/2
    output[:, [2,4]] -= (inp_dim - scaling_factor_h*im_dim_list[:,1].view(-1,1))/2
    
    output[:, [1,3]] /= scaling_factor_w
    output[:, [2,4]] /= scaling_factor_h

    # clip bounding box for those that extend outside the image
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i, 0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i, 1])
    
    number = sorted(
            [(classes[int(x[-1])], int(x[1])) for x in output],
            key = lambda x: x[1]
        )        
    
            
    return "".join([str(i[0]) for i in number])

def detect(img, img_size):
    img, img0 = data(img, img_size)
    img_tensor = torch.from_numpy(img).float()
    img_tensor /= 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    pred = plate_model(img_tensor, None)[0]
    pred = non_max_suppression(pred)

    img_c = img0.copy()
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img_c.shape).round()
            for *xyxy, conf, cls_ in reversed(det):
                coord = list(map(int, xyxy))
                c = int(cls_)
                y, x = int(coord[0]), int(coord[1])
                h = int(coord[2] - coord[0])
                w = int(coord[3] - coord[1])
                cropped = img0[x:x+w, y:y+h]
                number = number_recognition(cropped)
                
                if number and number not in seen_plates:
                    seen_plates.append(number)
                #elif not number: label = "license_plate"
                plot_one_box(xyxy, img_c, label=number, color=colors(c, True), line_thickness=2)
    return img_c

def main():
    global seen_plates, plate_model, number_model, num_classes, confidence, nms_thresh, classes, inp_dim, inp_dim2, CUDA, colors
    st.title("License Plate Recognition")
    st.header("Reads the character on license plate...")
    st.write(
            """ An application of computer vision to read and store the
                digits on the license plate of vehicles.

                Outcomes might be slow because inference is performed on CPU.
            """
            )
    st.write(
            """
            Upload the video file (mp4).
            """
            )
    video = st.file_uploader("Your video input.")
    if st.button("start recognition"):
        if not video: st.warning("no input video!!!")
        else:
            ms = datetime.now().microsecond
            tempf = tempfile.NamedTemporaryFile(delete=True, prefix=str(ms), dir=".", suffix=".mp4")
            tempf.write(video.read())
            seen_plates = []
            plate_model = torch.load('./best.pt')['model'].float()
            number_model = torch.load("./crnet.pt").eval()

            num_classes = 35
            confidence = 0.6
            nms_thresh = 0.4
            classes = load_classes("./lp-recognition.names")
            inp_dim = 128
            inp_dim2 = 352
            CUDA = torch.cuda.is_available()
    
            colors = Colors()

            #cap = cv2.VideoCapture("./video/video1.mp4")#"./images/michael/img/imgi/image_%04d.jpg"
            cap = cv2.VideoCapture(tempf.name)
            assert cap.isOpened(), "Unable to read from source..."
            stframe = st.empty()
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    pred = detect(img, (640, 640))
                else:
                    break
                img_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
                stframe.image(img_pred, width=720)
            cap.release()

            st.write("Seen plates:")
            st.write("\n".join(seen_plates))
    
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
