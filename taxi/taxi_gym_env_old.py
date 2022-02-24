#importing required libraries for gym env
import gym
import numpy as np
from visgrid.taxi.taxi import VisTaxi5x5
# from gym.envs.classic_control import rendering
#from stable_baselines.common.env_checker import check_env

#debugging
import pdb


'''Taxi Environment: inherits from gym.Env class'''
class TaxiEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self, num_passengers, greyscale_visualization, add_noise, randomize_positions, have_goal, ignore_rewards, max_steps_per_episode):

        self.taxi_env = VisTaxi5x5(n_passengers=num_passengers, grayscale=greyscale_visualization)

        #reset the env initially
        # get_state() in gridworld.py [GridWorld] returns agent's init. state
        # get_state() in taxi.py [BaseTaxi] returns agent's position + passengers position/in-taxi state
        # reset() from taxi.py [BaseTaxi] returns rendered env
        taxi_rendering = self.taxi_env.reset(goal=have_goal, explore=randomize_positions)
        # taxi_rendering = self._render_gridworld(taxi_rendering)

        #render the puzzle and measure height and width: this is our observation space
        (height,width,channels) = taxi_rendering.shape
        self.height = height; self.width = width



        '''choose either approach for state information: only agent position or passenger positions/in-taxi + taxi position'''
        self.agent_state = self.taxi_env.agent.position
        # self.agent_state = self.taxi_env.get_state()


        # action space between 0 to 4:  [base grid actions] 0: LEFT, 1: RIGHT, 2: UP, 3: DOWN, 4: PICKUP [added new in taxi.py]
        self.action_space = gym.spaces.Discrete(len(self.taxi_env.actions))



        #encoding the observation space as a grid of pixels (each in range 0-255)
        self.observation_space = gym.spaces.Box(low = 0, high = 255, shape=(height, width, channels), dtype=np.uint8)


        #tracking steps per episode + state of termination
        self.T = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.ignore_rewards = ignore_rewards




    def reset(self, goal=True, explore=True):
        '''resets the state to the starting point'''

        rendered_taxi = self.taxi_env.reset(goal,explore)
        # rendered_taxi = self._render_gridworld(rendered_taxi)
        self.T = 0

        #update the agent_state: taxi position
        '''choose either approach for state information: only agent position or passenger positions/in-taxi + taxi position'''
        self.agent_state = self.taxi_env.agent.position
        # self.agent_state = self.taxi_env.get_state()


        return rendered_taxi



    def step(self, action):
        '''
        converts an input from the self.action_space into a gridworld move, takes the action on the gridworld
        returns: state (as observation, after taking action), reward, done, info
        '''

        self.T += 1

        #note: the step() function accepts indexed actions from the self.action_space
        taxi_rendering, reward, done = self.taxi_env.step(action)


        if self.ignore_rewards:
            reward = 0.0
            done = False

        #temporary variable: information dict
        info = self._get_current_info()

        #checking reset condition
        timeout = self.T%self.max_steps_per_episode==0
        info['timeout'] = timeout
        info['is_terminal'] = done

        done = done or timeout

        return taxi_rendering, reward, done, info


    def render(self, mode='rgb_array', highlight = False, tile_size = 1, close=False):
        '''render the environment to the screen'''

        assert mode in ("human", "rgb_array"), mode

        #returns an array of size (height,width,3) with values 0-255
        if mode=='rgb_array':
            taxi_rendering = self.taxi_env.render()
            #taxi_rendering = self._render_gridworld(taxi_rendering)

        #physically display the rendering of the puzle
        elif mode=='human':
            pass


        return taxi_rendering

    def _get_action_space(self):
        '''return discrete action space for env'''

        return self.action_space

    def _get_current_info(self):
        '''return info dictionary for env'''

        info = {}

        #extract state information for info dict
        state_info = self.taxi_env.get_state()

        taxi_position = state_info[0:2];  info['taxi'] = taxi_position

        #store position + in-taxi information for each passenger
        for i in range(0,len(state_info[2:]),3):

            info['p{}'.format(i)] = state_info[i:i+3]

        return info

    def _render_gridworld(self, taxi_rgb_array, mode='rgb_array'):
        '''function to render taxi world using classic_control.rendering'''

        (height,width,channels) = taxi_rgb_array.shape

        scale_factor = (self.screen_height//height, self.screen_width//width)

        def gen_polygon(r,c):
            return rendering.FilledPolygon([  (c*scale_factor[1], r*scale_factor[0]),
                                            (c*scale_factor[1], (r+1)*scale_factor[0]),
                                            ((c+1)*scale_factor[1],(r+1)*scale_factor[0]),
                                            (c*scale_factor[1], (r+1)*scale_factor[0])
                                            ])

        #set up rendering if it doesn't exist
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

        self.viewer.window.clear()

        #populate viewer scrren with pixel values from rgb array
        for r in range(height):
            for c in range(width):

                pixel = get_polygon(r,c)

                pixel.set_color(tuple(taxi_rgb_array[r,c,:]/255))

                self.viewer.add_geom(block)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')
