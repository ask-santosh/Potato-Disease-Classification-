#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[13]:


IMAGE_SIZE =256
BATCH_SIZE =32 
EPOCHS = 50


# In[14]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle = True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE
    
)


# In[15]:


class_names = dataset.class_names
class_names
len(dataset)


# In[16]:


plt.figure(figsize=(14,14))
bg_color = 'white'
for image_batch, label_batch in dataset.take(1):
    for i in range(15):
        ax = plt.subplot(3,5,i+1)
        
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]], color=bg_color)
    


# In[40]:


def get_dataset_partitions_tf(ds, train_split=0.8, test_split=0.1,val_split=0.1, shuffle=True, shuffle_size=10000):
    ds_size = len(ds)
    print(ds_size)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(ds_size*train_split)
    print(train_size)
    val_size = int(ds_size*val_split)
    print(val_size)
    train_ds = ds.take(train_size)
    print(train_ds)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[41]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[42]:


print(len(test_ds))


# In[45]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[48]:


resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


# In[49]:


data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


# In[53]:


CHANNELS=3
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3
model = models.Sequential([resize_and_rescale,
                          data_augmentation,
                          layers.Conv2D(32,(3,3), activation='relu', input_shape=input_shape),
                          layers.MaxPooling2D((2, 2)),
                          layers.Conv2D(64,kernel_size = (3,3), activation='relu'),
                          layers.MaxPooling2D((2, 2)),
                          layers.Conv2D(64,kernel_size = (3,3), activation='relu'),
                          layers.MaxPooling2D((2, 2)),
                          layers.Conv2D(64, (3,3), activation='relu'),
                          layers.MaxPooling2D((2, 2)),
                          layers.Conv2D(64, (3,3), activation='relu'),
                          layers.MaxPooling2D((2, 2)),
                          layers.Conv2D(64, (3,3), activation='relu'),
                          layers.MaxPooling2D((2, 2)),
                          layers.Flatten(),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(n_classes, activation='softmax'),
                           
                          ])
model.build(input_shape= input_shape)


# In[55]:


model.summary()


# In[56]:


model.compile(
              optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics = ['accuracy']
)


# In[57]:


history = model.fit(train_ds,
                   epochs = EPOCHS,
                   batch_size= BATCH_SIZE,
                   verbose = 1,
                   validation_data = val_ds)


# In[58]:


scores = model.evaluate(test_ds)


# In[59]:


scores


# In[60]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[73]:


plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label="Training Loss")
plt.plot(range(EPOCHS), val_loss, label = 'Validation Loss')
plt.legend(loc='upper right')
plt.title("Training and Validation Loss")


# In[96]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("First image to predict")
    plt.imshow(first_image)
    print("actual label:", class_names[first_label])
    batch_prediction = model.predict(images_batch)
    print("predicted label:", class_names[np.argmax(batch_prediction[0])])


# In[109]:


def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence= round(100 * (np.max(predictions[0])),2)
    return predicted_class, confidence


# In[110]:


plt.figure(figsize=(15,15))
bg_color = 'white'
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"Actual class:{actual_class}, \n Predicted class:{predicted_class}, \n Confidence:{confidence}%",color=bg_color)
        plt.axis("off")


# In[118]:


import os
model.save(f"../models/1")


# In[ ]:




