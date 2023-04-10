import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [7.00, 3.50] # width, height
plt.rcParams["figure.autolayout"] = True # adjust subplot params so that the subplot(s) fits in to the figure area
columns = ["Frame", "Fall", "Fall Prob", "Fall Prob Threshold", "Time Stamp"] # columns to read from csv file
df = pd.read_csv("fall_data.csv", usecols=columns)  # read csv file

df["Time Stamp"] = pd.to_datetime(df["Time Stamp"])

# Calculate the time difference between consecutive frames
time_diff = df["Time Stamp"].diff()

# Define the time interval to group by (0.33 seconds)
interval = pd.Timedelta("0.33 seconds")

# Define a grouping function that groups by consecutive intervals
def group_by_interval(timestamp):
    base = timestamp.min()
    freq_nanos = interval.total_seconds() * 1_000_000_000
    base_nanos = base.value // 10 ** 9
    offset_nanos = int(base_nanos // freq_nanos * freq_nanos)
    offset = np.timedelta64(offset_nanos, 'ns')
    return pd.Grouper(key="Time Stamp", freq=interval, base=offset)

# Group the data by consecutive 0.33-second intervals
grouped = df.groupby(group_by_interval(df["Time Stamp"]))

# Print the size of each group (number of rows)
for name, group in grouped:
    print(f"Group {name}: {len(group)} rows")
    
#  save the groups to a list of dataframes and then plot them
# groups = [group for _, group in grouped]
# plt.plot(groups[0]["Frame"], groups[0]["Fall Prob"], label="Fall Prob", color="red" ) # plot fall probability

# grouped = df.groupby(df.index // 4)

# # Iterate over each group of 4 frames
# for i, group in grouped:
#     # Do something with the group of 4 frames
#     print(f"Group {i}:")
#     print(group)


# for i in range(len(df["Frame"])):
#     if i == 0:
#         continue
#     else:
#         if df["Frame"][i] - df["Frame"][i-1] > 1:
#             print("Frame", df["Frame"][i], "is not 1 frame after frame", df["Frame"][i-1])
#             # print("The time difference between these frames is", int(df["Time Stamp"][i]) - int(df["Time Stamp"][i-1]), "seconds")
#             print("The time difference between these frames should be 0.033 seconds")
#             print("")
#         else:
#             if pd.to_datetime(df["Time Stamp"][i]) - pd.to_datetime(df["Time Stamp"][i-1]) != 0.033:
#                 print("Frame", df["Frame"][i], "is not 1 frame after frame", df["Frame"][i-1])
#                 print("The time difference between these frames is",pd.to_datetime(df["Time Stamp"][i]) - pd.to_datetime(df["Time Stamp"][i-1]), "seconds")
#                 print("The time difference between these frames should be 0.033 seconds")
#                 print("")    
                
# print("Contents in csv file:", df)
plt.plot(df["Frame"], df["Fall Prob"], label="Fall Prob", color="red" ) # plot fall probability
plt.plot(df["Frame"], df["Fall"], label="Fell") # plot fall

"""
plt.plot(df["Frame"], df["Fall Prob Threshold"], label="Fall Prob Threshold") 
plot fall probability threshold defaults to 0.5 unless changed or error occurs
"""
plt.xlabel("Frame (1 frame = 0.033s)") # x-axis label
plt.show()