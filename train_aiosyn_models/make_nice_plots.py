import sys

import pandas as pd
import matplotlib.pyplot as plt

# File paths to CSV files
file_path_1 = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/result_csvs/GEN-880__training_train_dice_macro.csv"
file_path_2 = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/result_csvs/GEN-881__training_train_dice_macro.csv"
file_path_3 = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/result_csvs/GEN-878__training_train_dice_macro.csv"
file_path_4  ="/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/result_csvs/GEN-882__training_train_dice_macro.csv"
file_path_5 = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/result_csvs/GEN-879__training_train_dice_macro.csv"
file_path_6 = "/Users/Mees_1/MasterThesis/Aiosyn/code/ThesisProject/generationLDM/train_aiosyn_models/result_csvs/GEN-877__training_train_dice_macro.csv"

max_dices = [0.9886907339096068, 0.9870271682739258, 0.9840124249458312, 0.9777214527130128, 0.9671801328659058, 0.9822264909744264]
min_losses = [0.0700087025761604, 0.0582326650619506, 0.0577204972505569, 0.0616129823029041, 0.1142569929361343, 0.0457435473799705]
# Reading CSV files into dataframes
df1 = pd.read_csv(file_path_1)
df2 = pd.read_csv(file_path_2)
df3 = pd.read_csv(file_path_3)
df4 = pd.read_csv(file_path_4)
df5 = pd.read_csv(file_path_5)
df6 = pd.read_csv(file_path_6)

# Function to smooth data using a moving average
def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Assuming each CSV file has a column named 'dice'
# You may need to adjust this if your columns have different names
val_loss_1 = df1['dice']
val_loss_2 = df2['dice']
val_loss_3 = df3['dice']
val_loss_4 = df4['dice']
val_loss_5 = df5['dice']
val_loss_6 = df6['dice']

mins = []
mins.append(val_loss_1.min())
mins.append(val_loss_2.min())
mins.append(val_loss_3.min())
mins.append(val_loss_4.min())
mins.append(val_loss_5.min())
mins.append(val_loss_6.min())

window_size = 1 # Adjust the window size for smoothing as needed
val_loss_1_smooth = smooth_data(val_loss_1, window_size)
val_loss_2_smooth = smooth_data(val_loss_2, window_size)
val_loss_3_smooth = smooth_data(val_loss_3, window_size)
val_loss_4_smooth = smooth_data(val_loss_4, window_size)
val_loss_5_smooth = smooth_data(val_loss_5, window_size)
val_loss_6_smooth = smooth_data(val_loss_6, window_size)

# Plotting the smoothed validation loss data
plt.figure(figsize=(10, 6))

plt.plot(val_loss_1_smooth, label='100_0')
plt.plot(val_loss_2_smooth, label='75_25')
plt.plot(val_loss_3_smooth, label='50_50')
plt.plot(val_loss_4_smooth, label='75_25')
plt.plot(val_loss_5_smooth, label='0_100')
plt.plot(val_loss_6_smooth, label='100_100')


# Adding title and labels
fontsize = 18
plt.rcParams.update({'font.size': 18})
plt.title('DSC With Different Training Data')
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('DSC', fontsize=fontsize)

# Adding legend
plt.legend(fontsize = 20)

# Show plot
plt.show()
