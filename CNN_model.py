import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np


# Directory for images
full_data_dir = 'C:/{path}/re_im_norm_STFT'

# Load dataframes for the second case 
train_df = pd.read_pickle("train_df_case2.pkl")
test_df = pd.read_pickle("test_df_case2.pkl")

#Shuffle train_df data
train_df = train_df.sample(frac = 1)

### CNN model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.20)
test_datagen = ImageDataGenerator(rescale=1./255)


img_width = 84
img_height = 55 
batch_size = 816 

#Generators
train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory=full_data_dir,
                                                    x_col='ImageID',
                                                    y_col=['T1','T2'],
                                                    target_size=(img_width,img_height),
                                                    class_mode='raw',
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=42,
                                                    subset='training')

valid_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                    directory=full_data_dir,
                                                    x_col='ImageID',
                                                    y_col=['T1','T2'],
                                                    target_size=(img_width,img_height),
                                                    class_mode='raw',
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=42,
                                                    subset='validation')

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df,
                                                    directory=full_data_dir,
                                                    x_col='ImageID',
                                                    y_col=None,
                                                    target_size=(img_width,img_height),
                                                    class_mode=None,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    seed=42)

#Custom metrics for T1 and T2 accuracy
def cma1(y_train, y_pred):
    return 1-tf.math.reduce_mean(tf.math.abs(y_train[:,0]-y_pred[:,0])/(y_train[:,0]), axis=0, keepdims=False)
def cma2(y_train, y_pred):
    return 1-tf.math.reduce_mean(tf.math.abs(y_train[:,1]-y_pred[:,1])/(y_train[:,1]), axis=0, keepdims=False)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from tensorflow.keras.regularizers import l2
import time

start_time = time.time()

# VGG like model with the trend Conv-Conv-Pool-Conv-Conv-Pool

model = Sequential()
#1st Convolutional Layer                
model.add(Conv2D(filters = 32, kernel_size=(3,3), padding='same', input_shape=(img_width,img_height,3), activation='relu',
                 kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#2nd Convolutional Layer
model.add(Conv2D(filters = 32, kernel_size=(3,3), padding='same', activation='relu',
                 kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))

#3d Convolutional Layer
model.add(Conv2D(filters = 64, kernel_size=(3,3), padding='same', activation='relu',
                 kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))) #64
#4th Convolutional Layer
model.add(Conv2D(filters=64,kernel_size=(3,3), padding='same', activation='relu',
                 kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))

#5th Convolutional Layer
model.add(Conv2D(filters=128,kernel_size=(3,3), padding='same', activation='relu',
                 kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))

#Flatten out the model
model.add(Flatten())

#Add first Dense layer
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#Add second Dense layer
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))) 

#Output layer
model.add(Dense(2, activation='relu'))

#Optimizer - Case 1
opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 

#Optimizer - Case 2
opt = tf.keras.optimizers.Adam(learning_rate=0.0009) 

#Compile the model
model.compile(loss='mape', optimizer=opt, metrics = [cma1, cma2]) 

model.summary()

#Training the model
step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
step_size_test = test_generator.n//test_generator.batch_size


history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    validation_data=valid_generator,
                    validation_steps=step_size_valid,
                    epochs=200)

print('--- %s Seconds ---' % (time.time() - start_time))

#Save the trained model
model.save("CNN_trained_model.h5")


#Predict the output
test_generator.reset()
y_pred = model.predict_generator(test_generator,steps = len(test_generator), verbose=1)


#Epochs - Accuracy
T1_acc = history.history['cma1']
T2_acc = history.history['cma2']
epochs = range(200)
plt.plot(epochs, T1_acc, 'r', label = 'T1 accuracy (cma1)')
plt.plot(epochs, T2_acc, 'b', label = 'T2 accuracy (cma2)')
plt.title('T1 and T2 accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Epochs - Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#Correlation coefficient - R^2
corr_matrix1 = np.corrcoef(test_df.iloc[:,1], y_pred[:,0])
corr_xy1 = corr_matrix1[0,1]
r_squared1 = corr_xy1**2
print(r_squared1)

corr_matrix2 = np.corrcoef(test_df.iloc[:,2], y_pred[:,1])
corr_xy2 = corr_matrix2[0,1]
r_squared2 = corr_xy2**2
print(r_squared2)


#(T1)Reference values - Estimated values 
plt.plot(test_df.iloc[:,1], y_pred[:,0], 'm.', label='Predictions')
plt.plot(test_df.iloc[:,1], test_df.iloc[:,1], 'k-', label='Reference line')
plt.plot(r_squared1, label = 'R^2 = 0.903')
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
plt.legend(loc='upper left')
plt.show()


#(T2)Reference values - Estimated values 
plt.plot(test_df.iloc[:,2], y_pred[:,1], 'c.', label='Predictions')
plt.plot(test_df.iloc[:,2], test_df.iloc[:,2], 'k-', label='Reference line')
plt.plot(r_squared2, label = 'R^2 = 0.914')
plt.xlabel('Reference T1 (ms)')
plt.ylabel('Estimated T1 (ms)')
plt.legend(loc='upper left')
plt.show()
