import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from train_model import dice_score
import cv2


# load model and test data
unet_model = load_model('unet_model.h5')
test_X = np.load('data/test_X.npy')
test_y = np.save('data/test_y.npy')

# predict mask for test
pred = unet_model.predict(np.expand_dims(test_X[30], axis=0))

# show difference between original photo and mask prediction
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_X[30], cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(pred[0], cv2.COLOR_BGR2GRAY))

print(dice_score(test_y[30], pred[0]))
