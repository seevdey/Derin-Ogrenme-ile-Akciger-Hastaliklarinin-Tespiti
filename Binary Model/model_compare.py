from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model1 = load_model(('Model - 1/model1'))
model2 = load_model(('Model - 2/model2'))
model3 = load_model(('Model - 3/model3'))
model3_1 = load_model(('Model - 3.1/model3.1'))
model3_2 = load_model(('Model - 3.2/model3.2'))
model3_2_1 = load_model(('Model - 3.2.1/model3.2.1'))
model3_2_2 = load_model(('Model - 3.2.2/model3.2.2'))
model3_2_2_0_1 = load_model(('Model - 3.2.2.0.1/model3.2.2.0.1'))
model3_2_2_0_2 = load_model(('Model - 3.2.2.0.2/model3.2.2.0.2'))
model3_2_2_1 = load_model(('Model - 3.2.2.1/model3.2.2.1'))
model3_2_2_2 = load_model(('Model - 3.2.2.2/model3.2.2.2'))
model3_2_2_3 = load_model(('Model - 3.2.2.3/model3.2.2.3'))
model3_2_2_4 = load_model(('Model - 3.2.2.4/model3.2.2.4'))
model3_2_3 = load_model(('Model - 3.2.3/model3.2.3'))
model3_2_4 = load_model(('Model - 3.2.4/model3.2.4'))

tmodel1 = load_model(('Transfer Model - 1/tf_model1'))
tmodel2 = load_model(('Transfer Model - 2/tf_model2'))
tmodel3 = load_model(('Transfer Model - 3/tf_model3'))

batchSize = 50
imgSize = 224
test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('images/test', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary')

loss1, acc1 = model1.evaluate(test_set)
loss2, acc2 = model2.evaluate(test_set)
loss3, acc3 = model3.evaluate(test_set)
loss3_1, acc3_1 = model3_1.evaluate(test_set)
loss3_2, acc3_2 = model3_2.evaluate(test_set)
loss3_2_1, acc3_2_1 = model3_2_1.evaluate(test_set)
loss3_2_2, acc3_2_2 = model3_2_2.evaluate(test_set)
loss3_2_3, acc3_2_3 = model3_2_3.evaluate(test_set)
loss3_2_4, acc3_2_4 = model3_2_4.evaluate(test_set)
loss3_2_2_1, acc3_2_2_1 = model3_2_2_1.evaluate(test_set)
loss3_2_2_2, acc3_2_2_2 = model3_2_2_2.evaluate(test_set)
loss3_2_2_3, acc3_2_2_3 = model3_2_2_3.evaluate(test_set)
loss3_2_2_4, acc3_2_2_4 = model3_2_2_4.evaluate(test_set)
loss3_2_2_0_1, acc3_2_2_0_1 = model3_2_2_0_1.evaluate(test_set)
loss3_2_2_0_2, acc3_2_2_0_2 = model3_2_2_0_2.evaluate(test_set)

tloss1, tacc1 = tmodel1.evaluate(test_set)
tloss2, tacc2 = tmodel2.evaluate(test_set)
tloss3, tacc3 = tmodel3.evaluate(test_set)

fig, ax = plt.subplots()
ax.set_xlabel('Model', loc = 'right')
ax.set_ylabel('Accuracy', loc = 'top')
plt.title("Model - Accuracy")
x = np.array(["1", "2", "3", "3.1", "3.2", "3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.2.2.1", "3.2.2.2", "3.2.2.3", "3.2.2.4", "3.2.2.0.1", "3.2.2.0.2", "T 1", "T 2", "T 3"])
y = np.array([acc1, acc2, acc3, acc3_1, acc3_2, acc3_2_1, acc3_2_2, acc3_2_3, acc3_2_4, acc3_2_2_1, acc3_2_2_2, acc3_2_2_3, acc3_2_2_4, acc3_2_2_0_1, acc3_2_2_0_2, tacc1, tacc2, tacc3])
plt.figure(figsize=(25, 25))
plt.bar(x,y,color = "blue", width = 0.3)
plt.legend()
plt.savefig("model_acc_graph")

fig, ax = plt.subplots()
ax.set_xlabel('Model', loc = 'right')
ax.set_ylabel('Loss', loc = 'top')
plt.title("Model - Loss")
x = np.array(["1", "2", "3", "3.1", "3.2", "3.2.1", "3.2.2", "3.2.3", "3.2.4", "3.2.2.1", "3.2.2.2", "3.2.2.3", "3.2.2.4", "3.2.2.0.1", "3.2.2.0.2", "T 1", "T 2", "T 3"])
y = np.array([loss1, loss2, loss3, loss3_1, loss3_2, loss3_2_1, loss3_2_2, loss3_2_3, loss3_2_4, loss3_2_2_1, loss3_2_2_2, loss3_2_2_3, loss3_2_2_4, loss3_2_2_0_1, loss3_2_2_0_2, tloss1, tloss2, tloss3])
plt.figure(figsize=(25, 25))
plt.bar(x,y,color = "orange", width = 0.3)
plt.legend()
plt.savefig("model_loss_graph")