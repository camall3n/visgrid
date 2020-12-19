import gym
import logging
import sys
import time
import numpy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from tqdm import tqdm

from . import utils_for_q_learning, buffer_class
from gridworlds.nn import nnutils
from dmcontrol.markov import FeatureNet, build_phi_network

def rbf_function_on_action(centroid_locations, action, beta):
    '''
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x a_dim (action_size)]
    beta: float
        - Parameter for RBF function

    Description: Computes the RBF function given centroid_locations and one action
    '''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action.shape) == 2, "Must pass tensor with shape: [batch x a_dim]"

    diff_norm = centroid_locations - action.unsqueeze(dim=1).expand_as(centroid_locations)
    diff_norm = diff_norm**2
    diff_norm = torch.sum(diff_norm, dim=2)
    diff_norm = torch.sqrt(diff_norm + 1e-7)
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=1)  # batch x N
    return weights

def rbf_function(centroid_locations, action_set, beta):
    '''
    centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
    action_set: Tensor [batch x num_act x a_dim (action_size)]
        - Note: pass in num_act = 1 if you want a single action evaluated
    beta: float
        - Parameter for RBF function

    Description: Computes the RBF function given centroid_locations and some actions
    '''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action_set.shape) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"

    diff_norm = torch.cdist(centroid_locations, action_set, p=2)  # batch x N x num_act
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=2)  # batch x N x num_act
    return weights

class RBQFNet(nnutils.Network):
    def __init__(self, params, action_space, state_size):
        super().__init__()

        utils_for_q_learning.action_checker(action_space)
        self.params = params
        self.action_space = action_space
        self.action_size = len(action_space.low)
        self.state_size = state_size

        self.N = self.params['num_points']
        self.max_a = self.action_space.high[0]
        self.beta = self.params['temperature']

        self.value_module = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.N),
        )

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'],
                          self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'], self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        self.location_module[3].weight.data.uniform_(-.1, .1)
        self.location_module[3].bias.data.uniform_(-1., 1.)

        self.criterion = nn.MSELoss()

        self.params_dic = [{
            'params': self.value_module.parameters(),
            'lr': self.params['learning_rate']
        }, {
            'params': self.location_module.parameters(),
            'lr': self.params['learning_rate_location_side']
        }]
        try:
            if self.params['optimizer'] == 'RMSprop':
                self.optimizer = optim.RMSprop(self.params_dic)
            elif self.params['optimizer'] == 'Adam':
                self.optimizer = optim.Adam(self.params_dic)
            else:
                logging.warning('unknown optimizer ....')
        except:
            logging.warning("no optimizer specified ... ")

    def get_centroid_values(self, s):
        '''
        given a batch of s, get all centroid values, [batch x N]
        '''
        centroid_values = self.value_module(s)
        return centroid_values

    def get_centroid_locations(self, s):
        '''
        given a batch of s, get all centroid_locations, [batch x N x a_dim]
        '''
        centroid_locations = self.max_a * self.location_module(s)
        return centroid_locations

    def get_best_qvalue_and_action(self, s):
        '''
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        '''
        all_centroids = self.get_centroid_locations(s)
        values = self.get_centroid_values(s)
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]
        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
        # a -> all_centroids[idx] such that idx is max(dim=1) in allq
        # a = torch.gather(all_centroids, dim=1, index=indices)
        # (dim: bs x 1, dim: bs x action_dim)
        best, indices = allq.max(dim=1)
        if s.shape[0] == 1:
            index_star = indices.item()
            a = all_centroids[0, index_star]
            return best, a
        else:
            return best, None

    def forward(self, s, a):
        '''
        given a batch of s,a , compute Q(s,a) [batch x 1]
        '''
        centroid_values = self.get_centroid_values(s)  # [batch_dim x N]
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]
        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
        output = output.sum(1, keepdim=True)  # [batch x 1]
        return output

    def policy(self, s, epsilon, policy_noise=0):
        '''
        Given state s, at episode, take random action with p=eps
        Note - epsilon is determined by episode and whether training/testing
        '''
        if epsilon > 0 and random.random() < epsilon:
            a = self.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            with torch.no_grad():
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            if policy_noise > 0:
                noise = numpy.random.normal(loc=0.0, scale=policy_noise, size=len(a))
                a = a + noise
            return a

    def compute_loss(self, Q_target, s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix):
        Q_star, _ = Q_target.get_best_qvalue_and_action(sp_matrix)
        Q_star = Q_star.reshape((self.params['batch_size'], -1))
        with torch.no_grad():
            y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star
        y_hat = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat, y)
        return loss

class Agent:
    def __init__(self, params, env, device):
        self.params = params
        self.device = device
        self.buffer_object = buffer_class.buffer_class(max_length=params['max_buffer_size'],
                                                       env=env,
                                                       seed_number=params['seed_number'])

        s0 = env.reset()
        self.state_shape = s0.shape
        feature_type = self.params['features']
        if feature_type == 'expert':
            self.encoder = None
        elif feature_type == 'visual':
            self.encoder = build_phi_network(params, self.state_shape).to(device)
        elif feature_type == 'markov':
            self.encoder = FeatureNet(params, env.action_space, self.state_shape).to(device)
        if self.encoder is not None:
            print('Encoder:')
            print(self.encoder)
            print()

        s0_matrix = numpy.array(s0).reshape((1, ) + self.state_shape)
        z0 = self.encode(torch.as_tensor(s0_matrix).float().to(device))
        self.z_dim = len(z0.squeeze())

        self.Q_object = RBQFNet(params, env.action_space, self.z_dim).to(device)
        self.Q_object_target = RBQFNet(params, env.action_space, self.z_dim).to(device)
        self.Q_object_target.eval()
        print('Q-Network:')
        print(self.Q_object)
        print()

        utils_for_q_learning.sync_networks(target=self.Q_object_target,
                                           online=self.Q_object,
                                           alpha=params['target_network_learning_rate'],
                                           copy=True)

        policy_type = params['policy_type']
        if policy_type not in ['e_greedy', 'e_greedy_gaussian', 'gaussian']:
            raise NotImplementedError(
                'No get_action function configured for policy type {}'.format(policy_type))
        self.epsilon_schedule = lambda episode: 1.0 / numpy.power(
            episode, 1.0 / self.params['policy_parameter'])
        self.policy_noise = 0
        # override policy defaults for specific cases
        if policy_type == 'gaussian':
            self.epsilon_schedule = lambda episode: 0
        if policy_type in ['e_greedy_gaussian', 'gaussian']:
            self.policy_noise = self.params['noise']

    def encode(self, state):
        if self.encoder is None:
            return state
        return self.encoder(state)

    def act(self, s, episode, train_or_test):
        if train_or_test == 'train':
            epsilon = self.epsilon_schedule(episode)
            policy_noise = self.policy_noise
        else:
            epsilon = 0
            policy_noise = 0

        s_matrix = numpy.array(s).reshape((1, ) + self.state_shape)
        s = torch.from_numpy(s_matrix).float().to(self.device)
        z = self.encode(s)
        return self.Q_object.policy(z, epsilon, policy_noise)

    def update(self):
        if len(self.buffer_object) < self.params['batch_size']:
            return 0
        s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(
            self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix,
                              a_min=-self.params['reward_clip'],
                              a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        z_matrix = self.encode(s_matrix)
        zp_matrix = self.encode(sp_matrix)

        loss = self.Q_object.compute_loss(self.Q_object_target, z_matrix, a_matrix, r_matrix,
                                          done_matrix, zp_matrix)
        self.Q_object.zero_grad()
        loss.backward()
        self.Q_object.optimizer.step()
        utils_for_q_learning.sync_networks(target=self.Q_object_target,
                                           online=self.Q_object,
                                           alpha=self.params['target_network_learning_rate'],
                                           copy=False)
        return loss.cpu().data.numpy()

    def save(self):
        self.Q_object.save(name='Q_object', model_dir=self.params['models_dir'])
        if self.encoder is not None:
            self.encoder.save(name='encoder', model_dir=self.params['models_dir'])

class Trial:
    def __init__(self):
        params, env, device = self.parse_args()
        self.params = params
        self.env = env
        self.device = device

    @staticmethod
    def parse_args():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("Running on the GPU")
        else:
            device = torch.device("cpu")
            logging.info("Running on the CPU")
        hyper_parameter_name = sys.argv[1]
        alg = 'rbfdqn'
        params = utils_for_q_learning.get_hyper_parameters(hyper_parameter_name, alg)
        params['hyper_parameters_name'] = hyper_parameter_name
        params['alg'] = alg
        env = gym.make(params['env_name'])
        #env = gym.wrappers.Monitor(env, 'videos/'+params['env_name']+"/", video_callable=lambda episode_id: episode_id%10==0,force = True)
        params['seed_number'] = int(sys.argv[2])
        params['results_dir'] = 'dmcontrol/results/' + alg
        params['env'] = env
        return params, env, device

    def setup(self):
        utils_for_q_learning.set_random_seed(self.params)
        self.returns_list = []
        self.loss_list = []

    def teardown(self):
        pass

    def pre_episode(self, episode):
        logging.info("episode {}".format(episode))

    def run_episode(self, episode):
        s, done, t = self.env.reset(), False, 0
        while not done:
            a = self.agent.act(s, episode + 1, 'train')
            sp, r, done, _ = self.env.step(numpy.array(a))
            t = t + 1
            done_p = False if t == self.env.unwrapped._max_episode_steps else done
            self.agent.buffer_object.append(s, a, r, done_p, sp)
            s = sp

    def post_episode(self, episode):
        logging.debug('episode complete')
        #now update the Q network
        loss = []
        for count in tqdm(range(self.params['updates_per_episode'])):
            temp = self.agent.update()
            loss.append(temp)
        self.loss_list.append(numpy.mean(loss))

        self.every_n_episodes(10, self.evaluate_and_archive, episode)

    def evaluate_and_archive(self, episode, *args):
        temp = []
        for ep in range(10):
            s, G, done, t = self.env.reset(), 0, False, 0
            while done == False:
                a = self.agent.act(s, episode, 'test')
                sp, r, done, _ = self.env.step(numpy.array(a))
                s, G, t = sp, G + r, t + 1
            temp.append(G)
        logging.info("after {} episodes, learned policy collects {} average returns".format(
            episode, numpy.mean(temp)))
        self.returns_list.append(numpy.mean(temp))
        utils_for_q_learning.save(self.params['results_dir'], self.returns_list, self.loss_list,
                                  self.params)

    def every_n_episodes(self, n, callback, episode, *args):
        if (episode % n == 0) or (episode == self.params['max_episode'] - 1):
            callback(episode, *args)

    def run(self):
        self.setup()
        for episode in range(self.params['max_episode']):
            self.pre_episode(episode)
            self.run_episode(episode)
            self.post_episode(episode)
        self.teardown()

if __name__ == '__main__':
    trial = Trial()
    trial.run()
