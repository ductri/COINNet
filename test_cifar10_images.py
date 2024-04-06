
import numpy as np
from PIL import Image
import matplotlib.image

indices_selected=np.loadtxt('indices_selected.txt', dtype=int)
indices_not_selected=np.loadtxt('indices_not_selected.txt',dtype=int)
print(indices_selected)
print(indices_not_selected)

train_data=np.load('data/cifar10/train_images.npy')
print(np.shape(train_data))

train_data = np.reshape(train_data,(50000,32,32,3))
print(np.shape(train_data))

for i,ind in enumerate(indices_selected):
    #im = Image.fromarray(train_data[ind])
    #im.save("data/cifar10/instance_dep/IM_"+str(i)+".jpeg")
    im=np.transpose(np.reshape(train_data[ind],(3, 32,32)), (1,2,0))
    matplotlib.image.imsave('data/cifar10/instance_dep/IM_'+str(ind)+'.png', im)
    if i==100:
        break

for i,ind in enumerate(indices_not_selected[::-1]):
    im=np.transpose(np.reshape(train_data[ind],(3, 32,32)), (1,2,0))
    matplotlib.image.imsave('data/cifar10/instance_indep/IM_'+str(ind)+'.png', im)
    if i==100:
        break

