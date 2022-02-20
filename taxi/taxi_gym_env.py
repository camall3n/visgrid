#importing required libraries for gym env
import gym
import numpy as np
from taxi import VisTaxi5x5
#from stable_baselines.common.env_checker import check_env

#debugging
import pdb


'''Taxi Environment: inherits from gym.Env class'''
class TaxiEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self):


        kwargs = {n_passengers: 1}

        self.taxi_env = VisTaxi5x5(greyscale=False, kwargs)

        #reset the env initially
        # get_state() in gridworld.py [GridWorld] returns agent's init. state
        # get_state() in taxi.py [BaseTaxi] returns agent's position + passengers position/in-taxi state
        # reset() from taxi.py [BaseTaxi] returns rendered env
        taxi_rendering = self.taxi_env.reset(goal=True, explore=True)

        '''choose either approach for state information: only agent position or passenger positions/in-taxi + taxi position'''
        self.agent_state = self.taxi_env.agent.position
        self.agent_state = self.taxi_env.get_state()


        # action space between 0 to 4:  [base grid actions] 0: LEFT, 1: RIGHT, 2: UP, 3: DOWN, 4: PICKUP [added new in taxi.py]
        self.action_space = gym.spaces.Discrete(len(self.taxi_env.actions))


        #render the puzzle and measure height and width: this is our observation space
        (height,width,channels) = taxi_rendering.shape

        self.width = width; self.height = height

        #encoding the observation space as a grid of pixels (each in range 0-255)
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape=(height, width, channels), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({'image': self.observation_space})




    def reset(self, goal=True, explore=True):
        '''resets the state to the starting point'''

        rendered_taxi = self.taxi_env.reset(goal,explore)

        #update the agent_state: taxi position
        '''choose either approach for state information: only agent position or passenger positions/in-taxi + taxi position'''
        self.agent_state = self.taxi_env.agent.position
        self.agent_state = self.taxi_env.get_state()

        #store the observation/state representation after reset
        observation = {'image' : rendered_taxi}

        return observation



    def step(self, action):
        '''
        converts an input from the self.action_space into a gridworld move, takes the action on the gridworld
        returns: state (as observation, after taking action), reward, done, info
        '''

        #note: the step() function accepts indexed actions from the self.action_space
        taxi_rendering, reward, done = self.taxi_env.step(action)

        #temporary variable: information dict
        info = {}

        #extract state information for info dict
        state_info = self.taxi_env.get_state()

        taxi_position = state_info[0:2];  info['taxi'] = taxi_position

        #store position + in-taxi information for each passenger
        for i in range(0,len(state_info[2:]),3):

            info['p{}'.format(i)] = state_info[i:i+3]


        observation = {'image':taxi_rendering}

        return observation, reward, done, info


    def render(self, mode='rgb_array', highlight = False, tile_size = 1, close=False):
        '''render the environment to the screen'''

         #returns an array of size (height,width,3) with values 0-255
         taxi_rendering = self.taxi_env.render()

         #physically display the rendering of the puzle
         if mode=='human':
             pass 

         return taxi_rendering
