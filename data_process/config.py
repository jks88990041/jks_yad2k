import numpy as np
anchors =[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
anchors = np.array(anchors).reshape(-1, 2) #(5,2)
input_shape = (416, 416)
batch_size = 10
epochs = 100
VOCdevkit_path = 'F://train-data/VOCdevkit'
classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
           'chair','cow','diningtable','dog','horse','motorbike','person',
           'pottedplant','sheep','sofa','train','tvmonitor']
