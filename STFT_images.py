import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#Generated the STFT time-frequency map fingerprints

#Importing the dataset
re = pd.read_pickle("x_re.pickle")
im = pd.read_pickle("x_im.pickle")
re_im = np.concatenate((im, re), axis=1)
del re, im

#Add noise to the signal
sd = np.std(re_im)
noise = np.random.normal(0, sd*0.01, (re_im.shape)) 
re_im_noisy = re_im + noise

#Normalize the data
from sklearn import preprocessing
norm = preprocessing.Normalizer()
norm_signal = norm.transform(re_im_noisy) 

## Create the time-frequency map figures for each fingerprint (STFT)
for i in range(norm_signal.shape[0]):
   f, t, Zxx = signal.stft(norm_signal[i,:], fs = 1, window='hann', nperseg=300)
   image = plt.pcolormesh(t, f, np.abs(Zxx),cmap='nipy_spectral', shading='gouraud')
   plt.axis('off')
   plt.savefig("C:/{path}/re_im_norm_STFT/STFT_Fingerprint-%d.png" %i, bbox_inches = 'tight', pad_inches = 0)
   del image
   plt.cla()



