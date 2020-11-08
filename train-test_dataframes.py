from numpy import random
import pandas as pd
import os
import re

#Store image filenames in DataFrame
#Split into train - test DataFrames

### Case 2 directory
full_data_dir = 'C:/{path}/re_im_norm_STFT'

# Get the list of all files in the specified directory
full_set = os.listdir(full_data_dir)

# List the files in the Dataframe in the correct order
for_full_set = full_set.sort(key=lambda f:int(re.sub('\D', '', f)))

#Create dataframe
image_full_df = pd.DataFrame({'ImageID':full_set})

#Target values
Y = pd.read_pickle("y.pickle")
y_values = Y[:,0:2]

#Create dataframe
y_values_df = pd.DataFrame({'T1':y_values[:,0], 'T2':y_values[:,1]})

#Final dataframe (X_values and corresponding y_values)
train_df = pd.concat((image_full_df, y_values_df), axis=1)

#Initialise empty test_set dataframe
column_names = ["ImageID", "T1", "T2"]
test_df = pd.DataFrame(columns = column_names)

originalLength = len(train_df)

# Create test dataframe with 20% of train dataframe's random values
while len(test_df) < originalLength // 5:
    r = random.randint(0, originalLength - len(test_df) - 1)
    selectedRow = train_df.at[r, 'ImageID']
    #if selectedRow not in test_df.ImageID:
    test_df = test_df.append(train_df.iloc[r, 0:3], ignore_index = True)
    train_df.drop(r, inplace = True)
    train_df.reset_index(drop=True, inplace=True)

#Save Dataframes
train_df.to_pickle('C:/{path}/train_df_case2.pkl')
test_df.to_pickle('C:/{path}/test_df_case2.pkl')

