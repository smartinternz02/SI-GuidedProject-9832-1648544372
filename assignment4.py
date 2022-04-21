from keras.datasets import mnist
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  
import urllib
import gdown
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
#dataSet is downloaded from keras but was too large to submit on github 
df = pd.read_csv('/home/utkarsh/Desktop/SI-GuidedProject-9832-1648544372/dress.csv')
def show_image_from_url(image_url):
    response = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb), plt.axis('off')
plt.figure()
show_image_from_url(df['image_url'].loc[9564])
print('All categories : \n ', df['category'].unique())
n_classes = df['category'].nunique()
print('Total number of unique categories:', n_classes)
def image_processing(image_url):
    response = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image_hsv, (0,255,255), (0,255,255))
    if len(np.where(mask != 0)[0]) != 0:
        y1 = min(np.where(mask != 0)[0])
        y2 = max(np.where(mask != 0)[0])
    else:
        y1 = 0
        y2 = len(mask)
    if len(np.where(mask != 0)[1]) != 0:
        x1 = min(np.where(mask != 0)[1])
        x2 = max(np.where(mask != 0)[1])
    else:
        x1 = 0
        x2 = len(mask[0])
    image_cropped = image_gray[y1:y2, x1:x2]
    image_100x100 = cv2.resize(image_cropped, (100, 100))
    image_arr = image_100x100.flatten()
    return image_arr
url = 'https://drive.google.com/uc?id=1B6_rtcmGRy49hqpwoJT-_Ujnt6cYj5Ba'
output = 'X.npy'
gdown.download(url, output, quiet=False)
X = np.load('X.npy')
X[0:3]
np.random.seed(17)
for i in np.random.randint(0, len(X), 5):
  plt.figure()
  plt.imshow(X[i].reshape(100, 100)), plt.axis('off')
encoder = LabelEncoder()
Targets = encoder.fit_transform(df['category'])
Targets.shape
Y = to_categorical(Targets, num_classes = n_classes)
Y[0:3]
X_test = X[14000:,]
Y_test = Y[14000:,]
X_train, X_val, Y_train, Y_val = train_test_split(X[:14000,], Y[:14000,], test_size=0.15, random_state=13)
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(n_classes, activation='softmax'))
learning_rate = 0.001
model.compile(loss = categorical_crossentropy,optimizer = Adam(learning_rate),metrics=['accuracy'])
model.summary()
save_at = "/kaggle/working/model.hdf5"
save_best = ModelCheckpoint (save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
history = model.fit( X_train, Y_train, epochs = 15, batch_size = 100, callbacks=[save_best], verbose=1, validation_data = (X_val, Y_val))
plt.figure(figsize=(6, 5))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy', weight='bold', fontsize=16)
plt.ylabel('accuracy', weight='bold', fontsize=14)
plt.xlabel('epoch', weight='bold', fontsize=14)
plt.ylim(0.4, 0.9)
plt.xticks(weight='bold', fontsize=12)
plt.yticks(weight='bold', fontsize=12)
plt.legend(['train', 'val'], loc='upper left', prop={'size': 14})
plt.grid(color = 'y', linewidth='0.5')
plt.show()
model = load_model('/kaggle/working/model.hdf5')
score = model.evaluate(X_train, Y_train)
print('Accuracy over the test set: \n ', round((score[1]*100), 2), '%')
def find_similar_images(image_url, no_of_images):
    X_query = image_processing(image_url)
    X_query = X_query/255
    X_query = X_query.reshape(1, 100, 100, 1)
    Y_query = np.round(model.predict(X_query))
    i = np.where(Y_query == 1)[0].sum()
    print('Type detected by model:', encoder.classes_[i].upper())
    df_req = df.loc[ df['category'] == encoder.classes_[i]]
    df_req = df_req.reset_index(drop=True)

    if no_of_images > len(df_req):
        return(print('number of images needed are more than similar images in the dataset'))
    plt.figure()
    show_image_from_url(image_url)
    plt.title('Query Image')
    c = 1
    np.random.seed(13)
    for j in np.random.randint(0, len(df_req), no_of_images):
        plt.figure()
        url = df_req['image_url'].iloc[j]
        show_image_from_url(url)
        plt.title('Similar Image {}'.format(c))
        c += 1
find_similar_images('https://i.dailymail.co.uk/1s/2018/11/06/23/5855600-6360713-Ashley_James_stuns_in_emerald_green_animal_print_dress_at_glitzy-a-123_1541546195058.jpg', 5)