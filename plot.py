import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.xlabel('Time Elapsed')
plt.ylabel('Episode Reward Mean')

plt.grid(True)
plt.legend()

df = None

while plt.fignum_exists(plt.gcf().number):  # Check if the figure is still open
    try:
        df = pd.read_csv('logging/progress.csv')
    except:
        plt.pause(1)
        continue
    plt.clf()

    plt.plot(df['time/time_elapsed'], df['rollout/ep_rew_mean'], label='Reward', color='blue')
    plt.plot(df['time/time_elapsed'], df['rollout/ep_len_mean'], label='Total Timesteps', color='red')
    
    # plt.legend()

    plt.pause(1)

plt.show()
