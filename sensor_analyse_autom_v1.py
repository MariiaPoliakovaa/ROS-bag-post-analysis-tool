#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timedelta
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.records import Message, Schema
from mcap_ros2.reader import read_ros2_messages
import matplotlib.ticker as ticker
import os
import glob


# In[43]:


def time_window(stamps, start_ts, end_ts):
    """
    Returns all timestamps within the provided time window. This function is necessary because
    not all topics start sending at the same time, probably due to different starting times.

    Args:
        stamps (dict): A dictionary with the topic name as keys and list of timestamps as values
        start_ts (datetime): The start of the time window
        end_ts (datetime): The end of the time window
    Returns:
        dict[str, np.array(datetime)]: Dictionary with all timestamps in a certain window for each topic
    """
    window_stamps = {}
    for topic in stamps:
        # cast list of timestamps to np.array
        topic_stamps = np.array(stamps[topic])

        # use only timestamps within the provided window
        mask = (topic_stamps >= start_ts) & (topic_stamps < end_ts)
        window_stamps[topic] = topic_stamps[mask]

    return window_stamps


# In[44]:


# Define topics of interest
topics = ["/cam1", "/cam2", "/robin", "/falcon"]

# Find all .mcap files in the current directory
base_dir = os.getcwd()
mcap_files = glob.glob(os.path.join(base_dir, "*.mcap"))

if not mcap_files:
    raise FileNotFoundError("No .mcap files found in this directory!")

# Duration of the time window
window_duration = timedelta(seconds=20)

for mcap_file in mcap_files:
    print(f"\n📂 Processing file: {os.path.basename(mcap_file)}")

    stamps = {t: [] for t in topics}

    # Read messages
    with open(mcap_file, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, message, decoded_message in reader.iter_decoded_messages():
            if channel.topic in topics:
                seconds = decoded_message.header.stamp.sec
                nanoseconds = decoded_message.header.stamp.nanosec
                ts = datetime.fromtimestamp(seconds) + timedelta(microseconds=nanoseconds / 1000)
                stamps[channel.topic].append(ts)

    # Summarize & window each topic
    for topic, ts in stamps.items():
        print(f"\nTopic: {topic}")
        if not ts:
            print("  ⚠️ No messages found")
            continue

        start_time = ts[0]
        end_time = start_time + window_duration
        windowed_ts = [t for t in ts if start_time <= t <= end_time]

        # Length of the full message timeline
        total_length = ts[-1] - ts[0]

        print(f"  Total messages: {len(ts)}")
        print(f"  Full timeline length: {total_length}")
        print(f"  Window start:  {start_time}")
        print(f"  Window end:    {end_time}")
        print(f"  Window length: {window_duration}")
        #print(f"  Messages in 1s window: {len(windowed_ts)}")


# In[45]:


# frequencies at which messages are send per topic
for topic in stamps:
    timestamps = np.array(stamps[topic])
    freq = len(timestamps) / (timestamps.max() - timestamps.min()).total_seconds()
    print(topic, freq)


# In[46]:


start_ts = start_time
end_ts = end_time
window_stamps = time_window(stamps, start_ts, end_ts)


# In[47]:


fig, ax = plt.subplots(figsize=(8, 3))  # one axis, not a list

n = 5  # number of points to plot
colors = ['black', 'green', 'red', 'blue']

for i, topic in enumerate(('/cam1', '/cam2', '/robin', '/falcon')):
    y = np.ones(n) * i
    ax.scatter(window_stamps[topic][:n], y, color=colors[i], label=topic)

ax.legend()
ax.set_yticks(range(4))
ax.set_yticklabels(('/cam1', '/cam2', '/robin', '/falcon'))
ax.set_xlabel("Timestamps")
ax.set_title("Sensor Event Alignment")
  
ax.grid(True)
plt.show()


# In[48]:


fig, ax = plt.subplots(figsize=(20, 3))  # one axis

n = 30  # max number of points to plot
colors = ['black', 'green', 'red', 'blue']

for i, topic in enumerate(('/cam1', '/cam2', '/robin', '/falcon')):
    x = window_stamps[topic][:n]           # timestamps
    y = np.ones(len(x)) * i                # same length as x
    ax.scatter(x, y, color=colors[i], label=topic)

ax.legend()
ax.set_yticks(range(4))
ax.set_yticklabels(('/cam1', '/cam2', '/robin', '/falcon'))
ax.set_xlabel("Timestamps")
ax.set_title("Sensor Event Alignment")
ax.grid(True)
plt.show()



# In[49]:


for topic, ts_list in window_stamps.items():
    timestamps = np.array(ts_list)
    
    # Convert to milliseconds relative to the first message
    timestamps_ms = np.array([(t - timestamps[0]).total_seconds() * 1000 for t in timestamps])
    
    # Compute inter-message intervals in ms
    deltas = np.diff(timestamps_ms)
    
    # Basic stats
    mean_delta = np.mean(deltas) if len(deltas) > 0 else 0
    std_delta = np.std(deltas) if len(deltas) > 0 else 0
    freq = len(timestamps) / ((timestamps_ms[-1] - timestamps_ms[0])/1000) if len(timestamps) > 1 else 0  # Hz
    
    # CDF calculation
    counts, bin_edges = np.histogram(deltas, bins=30)
    cdf = np.cumsum(counts) / np.sum(counts) * 100
    
    # Percentiles
    percentiles = [90]
    percentile_values = []
    for p in percentiles:
        idx = np.searchsorted(cdf, p)
        interval_value = bin_edges[idx+1] if idx+1 < len(bin_edges) else bin_edges[-1]
        percentile_values.append(interval_value)
    
    # --- Create 3 plots in 1 row ---
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    
    # 1. Histogram of intervals
    if len(deltas) > 0:
        counts, bins, patches = ax[0].hist(deltas, bins=30, color='skyblue', edgecolor='black')
    else:
        ax[0].text(0.5, 0.5, "No data", ha='center', va='center', fontsize=14, color='red')
    ax[0].set_xticks(bins)
    ax[0].set_title("Interval Histogram")
    ax[0].set_xlabel("Interval (ms)")
    ax[0].set_ylabel("Count")
    ax[0].grid(True)
    ax[0].set_yscale('log')
    ax[0].text(0.5, -0.35, f"Mean = {mean_delta:.2f} ms\nStd = {std_delta:.2f} ms\nFreq = {freq:.2f} Hz\n90% <= {percentile_values[0]:.2f} ms",
               transform=ax[0].transAxes, ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    for label in ax[0].get_xticklabels():
       label.set_rotation(45)
    #print(f"{topic} - Histogram: Mean = {mean_delta:.2f} ms, Std = {std_delta:.2f} ms, Freq = {freq:.2f} Hz, 90% <= {percentile_values[0]:.2f} ms")

    
    # 2. Rolling frequency over 10 messages
    if len(deltas) >= 10:
        rolling_freq = 1000 / np.convolve(deltas, np.ones(10)/10, mode='valid')  # Hz
        ax[1].plot(rolling_freq, marker='o', linestyle='-')  # swapped axes
    else:
        ax[1].text(0.5, 0.5, "Not enough data", ha='center', va='center', fontsize=14, color='red')
    
    ax[1].set_title("Rolling Frequency (10 msg)")
    ax[1].set_xlabel("Messages")  # swapped label
    ax[1].set_ylabel("Freq (Hz)")  # swapped label
    ax[1].grid(True)
    ax[1].text(0.5, -0.35, f"Mean = {mean_delta:.2f} ms\nStd = {std_delta:.2f} ms\nAvg_Freq = {freq:.2f} Hz",
               transform=ax[1].transAxes, ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    ax[1].yaxis.set_inverted(True)
    
    if len(deltas) >= 10:
        print(f"{topic} - Rolling Frequency: Mean delta = {mean_delta:.2f} ms, Std delta = {std_delta:.2f} ms, Avg freq = {freq:.2f} Hz")
    else:
        print(f"{topic} - Rolling Frequency: Not enough data for rolling window")
    

    # 3. 
    n = 20 # number of messages to plot
    deltas_to_plot = deltas[:n]
    x = np.arange(len(deltas_to_plot))  # x length matches y
    ax[2].plot(x, deltas_to_plot)
    ax[2].set_xticks(x)
    ax[2].set_xlabel('Message index')
    ax[2].set_ylabel('Message offset (ms)')
    ax[2].grid(True)
    ax[2].set_title("Individual offsets")


    # Layout
    plt.suptitle(f"Sensor: {topic}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# In[50]:


fig, ax = plt.subplots(4, 1, figsize=(8, 15))

for i, (topic, ts_list) in enumerate(window_stamps.items()):
    timestamps = np.array(ts_list)
    
    # Convert to milliseconds relative to the first message
    timestamps_ms = np.array([(t - timestamps[0]).total_seconds() * 1000 for t in timestamps])
    
    # Compute inter-message intervals in ms
    deltas = np.diff(timestamps_ms)
    deltas = deltas[deltas < 60]
    
    # 1. Histogram of intervals
    counts, bins, patches = ax[i].hist(deltas, bins=30, color='skyblue', edgecolor='black')
    ax[i].set_xticks(bins)
    ax[i].set_title(f"Interval Histogram {topic}")
    ax[i].set_xlabel("Interval (ms)")
    ax[i].set_ylabel("Count")
    ax[i].grid(True)
    ax[i].set_yscale('log')
    for label in ax[i].get_xticklabels():
        label.set_rotation(45)

plt.tight_layout()


# In[ ]:





# ### Conclusion
# - cam1:
#     - The majority of offsets is between 32 and 37 ms (histogram). The individual offsets are mostly 31.999 and 32.0 (histogram and individual offsets)
#     - The offset is periodic, it seems to usualy be 31.999 and jump to 32.0 every 5 messages (individual offsets). There are also a few outliers with larger offsets as can be seen in the histogram
# - cam2:
#     - Similar to cam1 but the majority of offsets is between 31.998 and 31.9990. It is also periodic where each 6th message has an offset of 32.000
#  - robin: Robin is also periodic with a frequency of 2 messages. The offsets jump between ca 49.98 and 50.1.
#  - falcon: Falcon is not periodic. The offsets are roughly normal distributed around a mean of 50.17ms with a standard deviation of ca 4ms.
# 
# In general, the cameras are very similar with their delays. They both have delays around 32ms between each message with a periodically occuring, slightly larger, offset. This fits because both cameras operate at ca 30Hz frequency.
# 
# The lidars are differ from each other. Robin also has offsets which occur periodically slight variations but falcon is a lot more random.
# 
# I would conclude that the cameras and robin are rather stable and predictable (though care must be taken, because there are some unpredictable offsets occuring infrequently). Falcon is not as dependeble because we cannot assume that the next message will occur at a predictable timestamp.

# In[51]:


def timestep_deltas(timesteps1, timesteps2, threshold):
    """
    Computes the differences between two arrays of timesteps. Only
    uses timesteps where a match is found within the provided threshold.

    Computes timesteps2 - timesteps1 for each timestep where a match is found

    Args:
        timesteps1 (np.array(datetime)): The first array of timesteps
        timesteps2 (np.array(datetime)): The second array of timesteps
        threshold (int): The number of microseconds between two timesteps for them to be considered a match
    Returns:
        deltas (np.array(int)): The timestep deltas in microseconds
    """
    deltas = []
    for ts in timesteps1:
        # get the closes timestep in timesteps2 to ts
        diffs = timesteps2 - ts # Comupte the deltas between all points in timesteps2 and ts
        diffs = np.array([diff.microseconds for diff in diffs]) # Cast the deltas to microseconds

        # Check if the distance between ts and the closest point in timesteps2 is less than threshold
        abs_diffs = np.abs(diffs) # Use only absolute values, we dont care if the delta is negative
        min_idx = np.argmin(abs_diffs) # Get the index where the distance is minimal, this is the closest point to ts
        if abs_diffs[min_idx] < threshold: # Only consider the two points a match if the distance is less than threshold
            deltas.append(diffs[min_idx])
    return deltas


# In[52]:


# histogram of deltas for the cameras
threshold = 20000 #μS (microsiemens per centimeter, μS/cm)
cam_diffs = timestep_deltas(window_stamps['/cam1'], window_stamps['/cam2'], threshold)
cam_diffs = np.array(cam_diffs)/1000 # cast to miliseconds

plt.hist(cam_diffs)
plt.xlabel("Delta (ms)")
plt.title("Histogram of message-deltas between cam1 and cam2\nt(/cam2) - t(/cam1)")
plt.vlines([cam_diffs.mean()], 0, 250, color='red')
plt.show()


# In[53]:


# histogram of deltas for the cameras
threshold = 50000 #μS (microsiemens per centimeter, μS/cm)
cam_diffs = timestep_deltas(window_stamps['/cam1'], window_stamps['/falcon'], threshold)
cam_diffs = np.array(cam_diffs)/1000 # cast to miliseconds

plt.hist(cam_diffs)
plt.xlabel("Delta (ms)")
plt.title("Histogram of message-deltas between robin and falcon\nt(/falcon) - t(/cam1)")
plt.vlines([cam_diffs.mean()], 0, 150 , color='red')


# In[54]:


threshold = 50000 #μS (microsiemens per centimeter, μS/cm)
cam_diffs = timestep_deltas(window_stamps['/cam1'], window_stamps['/robin'], threshold)
cam_diffs = np.array(cam_diffs)/1000 # cast to miliseconds

plt.hist(cam_diffs)
plt.xlabel("Delta (ms)")
plt.title("Histogram of message-deltas between falcon and cam1\nt(/robin) - t(/cam1)")
plt.vlines([cam_diffs.mean()], 0, 150, color='red')


# In[55]:


# Define expected fps per topic
expected_fps = {
    '/cam1': 20,#20 fps * 60 s = 1200 frames
    '/cam2': 20,
    '/robin': 20,
    '/falcon': 20
}

# Calculate window length in seconds
window_stamps = time_window(stamps, start_ts, end_ts)
window_length = (end_ts - start_ts).total_seconds()

for topic in window_stamps:
    f_exp = expected_fps[topic]
    n_expected = f_exp * window_length
    n_actual = len(window_stamps[topic])
    loss_percent = (n_expected - n_actual) / n_expected * 100 
    print(f"{topic}: expected {n_expected:.0f}, actual {n_actual}, frame loss = {loss_percent:.2f}%")
    print(f"Window length = {window_length:.2f} s")


# In[56]:


# Parameters
n = 20  # number of points to plot

# Expected fps per topic
expected_fps = {
    '/cam1': 20,  # 20 fps
    '/cam2': 20,
    '/robin': 20,
    '/falcon': 20
}

# Use the actual window you already have
start_ts = datetime(2025, 10, 9, 14, 38, 59)
end_ts   = datetime(2025, 10, 9, 14, 39, 59)
window_length = (end_ts - start_ts).total_seconds()

# --- Cameras plot ---
camera_topics = ('/cam1', '/cam2')
colors_cam = ['black', 'green']

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
for i, topic in enumerate(camera_topics):
    topic_stamps = window_stamps.get(topic, [])
    if len(topic_stamps) == 0:
        continue

    y = np.ones(len(topic_stamps[-n:])) * i  # match y length to actual last n points
    ax.scatter(topic_stamps[-n:], y, color=colors_cam[i], label=topic)
    
    # Calculate frame loss based on actual data duration
    start_actual = topic_stamps[0]
    end_actual = topic_stamps[-1]
    window_actual = (end_actual - start_actual).total_seconds()
    n_expected = expected_fps[topic] * window_actual
    n_actual = len(topic_stamps)
    loss_percent = (n_expected - n_actual) / n_expected * 100

    # Display loss next to last point
    ax.text(topic_stamps[-1], i + 0.1, f"Loss: {loss_percent:.2f}%", 
            color=colors_cam[i], fontsize=9)

ax.set_yticks(range(len(camera_topics)))
ax.set_yticklabels(camera_topics)
ax.set_xlabel("Time")
ax.set_title(f"Camera sensors: Last {n} messages with frame loss %")
ax.grid(True)
ax.legend()
plt.show()

# --- LiDAR plot ---
lidar_topics = ('/robin', '/falcon')
colors_lidar = ['red', 'blue']

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
for i, topic in enumerate(lidar_topics):
    topic_stamps = window_stamps.get(topic, [])
    if len(topic_stamps) == 0:
        continue

    y = np.ones(len(topic_stamps[-n:])) * i
    ax.scatter(topic_stamps[-n:], y, color=colors_lidar[i], label=topic)
    
    # Calculate frame loss based on actual data duration
    start_actual = topic_stamps[0]
    end_actual = topic_stamps[-1]
    window_actual = (end_actual - start_actual).total_seconds()
    n_expected = expected_fps[topic] * window_actual
    n_actual = len(topic_stamps)
    loss_percent = (n_expected - n_actual) / n_expected * 100

    ax.text(topic_stamps[-1], i + 0.1, f"Loss: {loss_percent:.2f}%", 
            color=colors_lidar[i], fontsize=9)

ax.set_yticks(range(len(lidar_topics)))
ax.set_yticklabels(lidar_topics)
ax.set_xlabel("Time")
ax.set_title(f"LiDAR sensors: Last {n} messages with frame loss %")
ax.grid(True)
ax.legend()
plt.show()



# In[ ]:





# In[ ]:




