from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from nets.yololoss import yolo_loss
from data_process.config import anchors,classes,input_shape,batch_size
from nets.v2net import model_body
from data_process.data_loader import SequenceData
from keras.layers import Input, Lambda


input_image = Input(shape=(416, 416, 3))#输入图片为416*416，三通道（RGB）
boxes_input = Input(shape=(None, 5)) #表示一张图中所有目标信息
detectors_mask_input = Input(shape=(13, 13, 5, 1))# 目标掩码，确定目标位于哪一个单元格中的哪一个anchor
matching_boxes_input = Input(shape= (13, 13, 5, 5))# 目标在单元格中的anchor的编码位置和类别信息
model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': len(classes) })([model_body.output, boxes_input, detectors_mask_input, matching_boxes_input])

model = Model( [model_body.input, boxes_input, detectors_mask_input,matching_boxes_input],  model_loss )  # 将loss layer加入模型中
#配置模型
model.compile(
        optimizer=Adam(learning_rate = 0.0001), loss={
            'yolo_loss': lambda y_true, y_pred: y_pred })
model_body.load_weights("./ckpt.h5",by_name=True,skip_mismatch=True)

#训练模型
logging = TensorBoard(log_dir='logs/') #指定训练log目录
checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
                             monitor='loss', save_weights_only=True, save_best_only=True, period=3)  #保存模型

# 创建训练和验证数据集
train_sequence = SequenceData("data_process/2007_train.txt", input_shape, batch_size, anchors, len(classes))
val_sequence = SequenceData("data_process/2007_val.txt", input_shape, batch_size, anchors, len(classes))
#训练
model.fit_generator(train_sequence,
                    steps_per_epoch=train_sequence.get_epochs(),
                    validation_data=val_sequence,
                    validation_steps=val_sequence.get_epochs(),
                    validation_freq=10,
                    initial_epoch=0,
                    epochs=100,
                    workers=0,
                    callbacks=[checkpoint,logging])

