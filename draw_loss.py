#绘制训练loss，先把训练结果手动保存在了 train_log.txt中
import matplotlib.pyplot as plt
x = []
y = []
with open("./train_log.txt","r",encoding="utf-8") as f:
    lline = f.readlines()
    index =0
    for ll in lline:
        if ll.find("step - loss:") >= 0  :
            x.append(index)
            index += 1
            value = ll.split("step - loss:")[-1]
            if value.find("- val_loss: ")>0:
                value = value.split("- val_loss:")[0].strip()
            else:
                value = value.strip()
            y.append(float(value))
plt.title('train loss')
plt.plot(x , y , color='green', label='training loss')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()