import numpy as np
import sys,os
import cv2

#set your caffe root such as '/home/work/caffe_ssd/'
caffe_root = '/home/work/caffe_ssd/'
sys.path.insert(0, caffe_root + 'python')
import caffe


net_file= 'model/voc/MobileNetSSD_deploy.prototxt'
caffe_model='model/voc/MobileNetSSD_deploy.caffemodel'

test_dir = "image"
out_dir = "result"

isVito = True

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()

print ('caffe net')    
net = caffe.Net(net_file,caffe_model,caffe.TEST)

if isVito == True:
  CLASSES = ('background','C-fist', 'O-palm', 'P-photo', 'Dynamics')         
else:
  CLASSES = ('background',
             'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor')  

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile, img_code):
    print('imgfile=', imgfile, 'img_code=', img_code)
    origimg = cv2.imread(imgfile)
    img = preprocess(origimg)

    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()
    box, conf, cls = postprocess(origimg, out)

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])
       if (box[i][0]<0 or box[i][1]<0 or box[i][2]<0 or box[i][3]<0):
          continue
       
       if conf[i] < 0.5:
          continue
          
       cv2.rectangle(origimg, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       print('i=',i,'p1=',p1,'p2=',p2,'title=',title)
       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    #cv2.imshow("SSD", origimg)
    cv2.imwrite(out_dir+"/"+str(img_code)+".jpg", origimg)

    #k = cv2.waitKey(0) & 0xff    
    #Exit if ESC pressed
    #if k == 27 : return False
    return True

i = 0
for f in os.listdir(test_dir):
    i += 1
    if detect(test_dir + "/" + f, i) == False:
       break

print ('end')

