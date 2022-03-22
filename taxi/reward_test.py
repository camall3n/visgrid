from visgrid.taxi.taxi_gym_env import TaxiEnv
import pdb
import gym
import gym_minigrid
import pandas as pd
import os
from PIL import Image

#create Taxi Env
taxi_env = TaxiEnv(max_steps_per_episode = 50)

#create dataframe of transitions, rewards and actions
dataframe = pd.DataFrame(columns = ['state','next_state','action','reward', 'terminal','timeout'])

#sample random NUM_STEPS actions + reset with 50-step timeout
NUM_STEPS = 15000
SAVE_PATH = os.path.relpath('.')

if not os.path.isdir(os.path.join(SAVE_PATH,'images')):
    os.mkdir(os.path.join(SAVE_PATH,'images'))

count = 1
for i in range(1, NUM_STEPS+1):
    print("Step {} of {}".format(i, NUM_STEPS))
    #pdb.set_trace()
    #step 0: save current state image
    image_path = os.path.join(SAVE_PATH, 'images', 'state_{}.png'.format(count))
    count+=1
    observation = taxi_env.observation_space

    observation = Image.fromarray(observation)
    observation.save(image_path)

    #step 1: sample random action from environment
    random_action = taxi_env.action_space.sample()


    #step 2: take random action + record (s,s',a,r) in pandas dataframe
    _____, reward, terminal, info = taxi_env.step(random_action)
    next_image_path  = os.path.join(SAVE_PATH,'images', 'state_{}.png'.format(count))
    count+=1
    observation = taxi_env.observation_space
    
    observation = Image.fromarray(observation)
    observation.save(next_image_path)

    data_dict = {'state': image_path, 'next_state':next_image_path, 'action':random_action, 'reward':reward, 'terminal': info['is_terminal'],'timeout':info['timeout']}
    dataframe = dataframe.append(data_dict, ignore_index = True)

    
    taxi_env.reset()
    print("reward: ", reward)

print("Saving csv file ....")
#save dataframe as CSV data file
csv_path = os.path.join(SAVE_PATH, 'transition_buffer.csv')
dataframe.to_csv(csv_path)
