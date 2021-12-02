import six
from PIL import Image
import numpy as np
from PIL import Image,ImageFont, ImageDraw
lines = []
with open("2007_train.txt", "r") as f:
    lines = f.readlines()
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
           'chair','cow','diningtable','dog','horse','motorbike','person',
           'pottedplant','sheep','sofa','train','tvmonitor']


for line in lines:
    img_name = line.split(" ")[:1][0]
    print(img_name)
    infos = line.split(" ")[1:]
    img = Image.open(img_name )

    font = ImageFont.truetype(font='simhei.ttf',
                              size=np.floor(3e-2 * np.shape(img)[1] + 0.5).astype('int32'))
    draw = ImageDraw.Draw(img)
    for info in infos:
        xmin, ymin, xmax, ymax,cls_index = info.split(",")
        int_clas_index = int(cls_index)
        label = classes[int_clas_index]
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        if int(ymin) - label_size[1] >= 0:
            text_origin = np.array([int(xmin), int(ymin) - label_size[1]])
        else:
            text_origin = np.array([int(xmin), int(ymin) + 1])
        for i in range(1):
            draw.rectangle(
                [int(xmin)+i,int(ymin)+i,int(xmax)-i,int(ymax)-i],
                outline =(0,0,255))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(255,0,255))
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
    img.show()

