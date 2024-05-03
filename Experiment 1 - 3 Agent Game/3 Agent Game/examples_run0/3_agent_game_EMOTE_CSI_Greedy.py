
import sys
from typing import Deque

import random
import argparse
#from numpy.random import shuffle
sys.path.append("..")
import numpy as np
import marlgrid
import gym

from models import *

import matplotlib.pyplot as plt

#from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
#from marlgrid.envs import env_from_config
from marlgrid.envs import SympathyMultiGrid

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default="./logs/", type=str, nargs='?', help='where to store logs')
parser.add_argument('--save_dir', default="./saves/", type=str, nargs='?', help='where to store saves')
parser.add_argument('--gamma', default=0.5, type=float, nargs='?', help='gamma_weighting')
parser.add_argument('--empathy_mode', default="false", type=str, nargs='?', help='empathy_mode')
parser.add_argument('--CSI_mode', default="false", type=str, nargs='?', help='Run CSI')
parser.add_argument('--model_type', default="pixel", type=str, nargs='?', help='empathy model type')
parser.add_argument('--bench_mark', default=0, type=int, nargs='?', help='Bench Mark, 0 = off')
parser.add_argument('--training_mode', default="true", type=str, nargs='?', help='training mode')
args = parser.parse_args()

log_dir = args.log_dir
save_dir = args.save_dir
gamma = args.gamma
empathy_mode = args.empathy_mode
CSI_mode = args.CSI_mode
model_type = args.model_type
bench_mark = args.bench_mark
training_mode = args.training_mode

class TestRLAgent:
    def __init__(self, sympathetic_mode, empathetic_mode, Ghost_model_type, loss1_weight = 0.5, bench_mark = 0, training_mode = True, load_file = None):
        #self.player_window = InteractivePlayerWindow(
        #    caption="interactive marlgrid"
        #)
        self.episode_count = 0
        self.replay_memory = Deque()
        self.human_replay_memory = {}
        self.human_replay_memory['human'] = Deque()
        self.human_replay_memory['adversary'] = Deque()
        self.IRL_memory = {}
        self.IRL_memory['human'] = Deque()
        self.IRL_memory['adversary'] = Deque()

        # Sympathetic and empathetic models
        self.sympathetic_mode = sympathetic_mode
        self.empathetic_mode = empathetic_mode

        self.Ghost_model_type = Ghost_model_type

        self.params = {
            'save_file': 'save',
            'save_interval': 100,
            'keep_interval': 1000, 

            'training_start': 30, #30,

            'batch_size': 16,
            'view_window_width': 5,
            'view_window_height': 5,
            'loss1_weight': loss1_weight,

            'num_actions': 5,
            'num_human_actions': 5,

            'target_update': 2,
            'memory_limit': 10000,

            'no_episodes': 2000,
            'no_tests': 25,
            'discount': 0.9,
            'eps_initial': 1, 
            'eps_decay': 0.99, 

            'training_network_frac': 1.0,
            'recreate_error': 0.0005,
            'batch_multiple': 5


        }

        # Load trained file number 
        self.params['load_file'] = load_file

        if training_mode == False:
            self.params['save_dir'] = save_dir

        self.pixel_scale = 65
        # Initialise value function models
        self.q_net = DQN(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'], self.params)
        self.q_net_target = DQN(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'], self.params)
        
        self.params['load_file'] = None
        self.q_net_greedy = DQN_Greedy(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'])
        self.q_net_greedy_target = DQN_Greedy(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'])
        
        self.bench_mark = bench_mark

        # Dictionary for other (human and adversary)
        self.q_net_other = {}

        if bench_mark == 0:
            if self.empathetic_mode:
                if Ghost_model_type == 'pixel':
                    self.q_net_other['human'] =  DQN_Human_pixel(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'], self.params)
                    self.q_net_other['adversary'] = DQN_Adversary_pixel(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'], self.params)
                else:
                    self.q_net_other['human'] =  DQN_Human_state(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'], self.params)
                    self.q_net_other['adversary'] =  DQN_Adversary_state(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'], self.params)
            else:
                self.q_net_other['human'] =  DQN_Human_sympathetic(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'])
                self.q_net_other['adversary'] =  DQN_Human_sympathetic(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'])
        else:
            self.q_net_other['human'] =  DQN_Human_benchmark(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'], self.params)
            self.q_net_other['adversary'] =  DQN_Adversary_benchmark(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'], self.params)
    
        if self.empathetic_mode:
            if self.params['load_file'] is None:
                self.q_net_other['human'].Greedy_model.set_weights(self.q_net_greedy_target.model.get_weights())
                self.q_net_other['adversary'].Greedy_model.set_weights(self.q_net_greedy_target.model.get_weights())

        if Ghost_model_type == 'pixel':    
            self.stateRecreate = ReCreatePixel(self.params['view_window_width'],self.params['view_window_height'],self.params)
        elif Ghost_model_type == 'state':
            self.stateRecreate = ReCreateState(self.params['view_window_width'],self.params['view_window_height'],self.params)
            
        self.nextRobotstate = nextStateModel(self.params['view_window_width'],self.params['view_window_height'])
        self.nextHumanstate = nextStateModel(self.params['view_window_width'],self.params['view_window_height'])

        # Initial predicted human rewards
        self.other_rewards = {}
        self.other_rewards['human'] = np.array([0, 0, 0, 0])
        self.other_rewards['adversary'] = np.array([0, 0, 0, 0])
        self.greedy_rewards = np.array([0, 0, 0, 0])
        self.robot_rewards_pred = np.array([0, 0, 0, 0])
        self.robot_reward_vector = np.array([0, 0, 10, -1])
        # Reward features:
        # human_pellet, adversary_pellet, robot_pellet, win_game, step

        # Whether running training or testing mode
        self.training_mode = training_mode
        self.start_train_other =  False
        self.recreated_state_error = np.inf

        # Initialise parameters
        self.cost_total = 0
        self.greedy_cost_total = 0
        self.cnt = 0
        self.human_cost_total = 0
        self.human_perc_correct = 0
        self.human_cost_before = 0
        self.cnt_human = 0
        self.IRL_error = 0
        self.robot_nextstate_error = 0
        self.human_nextstate_error = 0

        self.beta_step = 0
        self.beta_step_cnt = 0
        self.beta_door_value = 0
        self.beta_door_cnt = 0
        self.beta_foodh = 0
        self.beta_foodh_cnt = 0
        self.beta_fooda = 0
        self.beta_fooda_cnt = 0
        self.beta_food = 0
        self.beta_food_cnt = 0
        self.beta_win = 0
        self.beta_win_cnt = 0
        self.beta_killed = 0
        self.beta_killed_cnt = 0
        self.beta_harmH = 0
        self.beta_harmH_cnt = 0

        self.reward_foodh = 0
        self.reward_fooda = 0
        self.reward_food = 0
        self.reward_door = 0
        self.reward_win = 0
        self.reward_step = 0
        self.reward_killed = 0
        self.reward_harmH = 0
        self.reward_foodh_pnext = 0
        self.reward_fooda_pnext = 0
        self.reward_food_pnext = 0
        self.reward_door_pnext = 0
        self.reward_win_pnext = 0
        self.reward_killed_pnext = 0
        self.reward_harmH_pnext = 0
        self.reward_step_pnext = 0

    def action_step(self, obs, agent0_pos, agent1_pos, agent2_pos, otherfood_pos, advfood_pos, walls, door_status, agent_no):

        if agent_no == 0:

            obs = np.array(obs)/self.pixel_scale
            obs = rgb2gray(obs).reshape(1,self.params['view_window_width'],self.params['view_window_height'],1)

            # ROBOT ACTIONS
            if np.random.uniform(0,1) > self.epsilon:
                scaled_agent0_pos = agent0_pos/np.array([self.grid_width, self.grid_height])
                scaled_agent1_pos = agent1_pos/np.array([self.grid_width, self.grid_height])
                scaled_agent2_pos = agent2_pos/np.array([self.grid_width, self.grid_height])
                
                I_door = np.array([int(door_status)]).reshape(1,1)
                I_a0_x = np.array([scaled_agent0_pos[0]]).reshape(1,1)
                I_a0_y = np.array([scaled_agent0_pos[1]]).reshape(1,1)
                I_a1_x = np.array([scaled_agent1_pos[0]]).reshape(1,1)
                I_a1_y = np.array([scaled_agent1_pos[1]]).reshape(1,1)
                I_a2_x = np.array([scaled_agent2_pos[0]]).reshape(1,1)
                I_a2_y = np.array([scaled_agent2_pos[1]]).reshape(1,1)

                if ((self.empathetic_mode == False) & (self.sympathetic_mode == False)):
                    # When training, always select action based on greedy model
                    actions_robot = self.q_net.model.predict([obs, I_door, I_a0_x, I_a0_y, I_a1_x, I_a1_y, I_a2_x, I_a2_y])[0]
                else:

                    if self.training_mode:
                        if np.random.uniform(0,1) > self.params['training_network_frac']:
                            # When testing, always select action based on sympathetic/empathetic model
                            actions_robot = self.q_net_greedy.model.predict([obs, I_door, I_a0_x, I_a0_y, I_a1_x, I_a1_y, I_a2_x, I_a2_y])[0]
                        else:
                            # When testing, always select action based on sympathetic/empathetic model
                            actions_robot = self.q_net.model.predict([obs, I_door, I_a0_x, I_a0_y, I_a1_x, I_a1_y, I_a2_x, I_a2_y])[0]
                    else:
                        # When training, always select action based on greedy model
                        actions_robot = self.q_net.model.predict([obs, I_door, I_a0_x, I_a0_y, I_a1_x, I_a1_y, I_a2_x, I_a2_y])[0]
                
                possible_actions = self.possibleActions(agent0_pos, walls, door_status, agent1_pos)
                actions_robot_possible = actions_robot[possible_actions]
                action_array = (possible_actions[np.argmax(actions_robot_possible)],6,6)

            else:

                possible_actions = self.possibleActions(agent0_pos, walls, door_status, agent1_pos)
                action_array = (np.random.choice(possible_actions),6,6)

        elif agent_no == 1:

            # Add epsilon to human movements
            if (np.random.uniform(0,1) < 0.2):
                possible_actions = self.possibleActions_Other(agent1_pos, walls, door_status)
                action_array = (6,np.random.choice(possible_actions),6)
            elif (len(otherfood_pos) == 0):
                action_array = (6,4,6)
            else:
                # Human action: Got forward if movement minimises distance to closest food.
                if (door_status):
                    if (np.array([7,1]) == otherfood_pos).all(1).any():
                         otherfood_pos_without_room = [otherfood_pos[-1]]
                    else:
                        otherfood_pos_without_room = otherfood_pos.copy()
                
                elif (len(otherfood_pos)==1):
                    otherfood_pos_without_room = otherfood_pos.copy()

                else:
                    otherfood_pos_without_room = otherfood_pos[:-1].copy() 

                possible_actions = self.possibleActions_Other(agent1_pos, walls, door_status)

                dist_to_food = []
                for a in possible_actions:
                    agent_tmp = agent1_pos.copy()

                    if a == 3:
                        agent_tmp[1] -= 1
                    if a == 0:
                        agent_tmp[0] += 1
                    if a == 1:
                        agent_tmp[1] += 1
                    if a == 2:
                        agent_tmp[0] -= 1


                    cur_dist = self.closestFood(agent1_pos, otherfood_pos_without_room, walls)
                    fwd_dist = self.closestFood(np.array(agent_tmp), otherfood_pos_without_room, walls)

                    dist_to_food.append(fwd_dist)

                I = np.argmin(dist_to_food)
                action_array = (6,possible_actions[I],6)

        elif agent_no == 2:

            # Add epsilon to human movements
            if (np.random.uniform(0,1) < 0.2):
                possible_actions = self.possibleActions_Other(agent2_pos, walls, door_status)
                action_array = (6,6,np.random.choice(possible_actions))
            elif (len(advfood_pos) == 0):
                action_array = (6,6,4)
            else:
                # Human action: Got forward if movement minimises distance to closest food.
                if (door_status):
                    if (np.array([7,1]) == advfood_pos).all(1).any():
                         otherfood_pos_without_room = [advfood_pos[-1]]
                    else:
                        otherfood_pos_without_room = advfood_pos.copy()
                
                elif (len(advfood_pos)==1):
                    otherfood_pos_without_room = advfood_pos.copy()

                else:
                    otherfood_pos_without_room = advfood_pos[:-1].copy() 

                possible_actions = self.possibleActions_Other(agent2_pos, walls, door_status)

                dist_to_food = []
                for a in possible_actions:
                    agent_tmp = agent2_pos.copy()

                    if a == 3:
                        agent_tmp[1] -= 1
                    if a == 0:
                        agent_tmp[0] += 1
                    if a == 1:
                        agent_tmp[1] += 1
                    if a == 2:
                        agent_tmp[0] -= 1


                    cur_dist = self.closestFood(agent2_pos, otherfood_pos_without_room, walls)
                    fwd_dist = self.closestFood(np.array(agent_tmp), otherfood_pos_without_room, walls)

                    dist_to_food.append(fwd_dist)

                I = np.argmin(dist_to_food)
                action_array = (6,6,possible_actions[I])


        return action_array

    def possibleActions_Other(self, agent_pos, walls_original, door_status):

        walls = walls_original.copy()

        #if not door_status:
        #    walls.append([7,2])

        directions = [4] # Can always stop

        # Check if can go up
        for dir in [0,1,2,3]:
            agent_tmp = agent_pos.copy()

            if dir == 3:
                agent_tmp[1] -= 1
            if dir == 0:
                agent_tmp[0] += 1
            if dir == 1:
                agent_tmp[1] += 1
            if dir == 2:
                agent_tmp[0] -= 1

            if (agent_tmp[0] > 0) and (agent_tmp[1] > 0) and (agent_tmp[0] < (self.grid_width-1)) and (agent_tmp[1] < (self.grid_height-1)):
                #if not (np.array(agent_tmp) == walls).all(1).any():
                directions.append(dir)

        return directions

    def possibleActions(self, agent_pos, walls_original, door_status, agent1_pos):

        walls = walls_original.copy()

        #if not door_status:
        #    walls.append([7,2])

        directions = [4] # Can always stop

        # Check if can go up
        for dir in [0,1,2,3]:
            agent_tmp = agent_pos.copy()

            if dir == 3:
                agent_tmp[1] -= 1
            if dir == 0:
                agent_tmp[0] += 1
            if dir == 1:
                agent_tmp[1] += 1
            if dir == 2:
                agent_tmp[0] -= 1

            if (agent_tmp[0] > 0) and (agent_tmp[1] > 0) and (agent_tmp[0] < (self.grid_width-1)) and (agent_tmp[1] < (self.grid_height-1)):
                #if not (np.array(agent_tmp) == walls).all(1).any():
                directions.append(dir)

        # Ability to toggle open door
        #if (agent_pos == np.array([7,3])).all() and (not door_status):
        #    directions.append(4) 

        return directions

    def closestFood(self, pos, food, walls):
        """
        closestFood -- this is similar to the function that we have
        worked on in the search project; here its all in one place
        """
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if any(np.array([(np.array([pos_x,pos_y]) == foods).all() for foods in food])):
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = self.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def getLegalNeighbors(self, position, walls):
        # Directions
        directions = {'NORTH': (0, 1),
        'SOUTH': (0, -1),
        'EAST':  (1, 0),
        'WEST':  (-1, 0),
        'STOP':  (0, 0)}

        directionsAsList = directions.items()

        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == self.grid_width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == self.grid_height: continue
            if not any(np.array([(np.array([next_x,next_y]) == wall).all() for wall in walls])): neighbors.append((next_x, next_y))
        return neighbors

    def save_step(self, obs_1, act, rew, obs_2, obs_3, done, reward_vector, door, door_ns, obs_2_door,won):
        #print(f"   step {self.step_count:<4d}: reward {rew} (episode total {self.cumulative_reward})")

        experience = [obs_1, act, rew, obs_3, done, reward_vector, door, door_ns, obs_2, int(obs_2_door),won]

        self.replay_memory.append(experience)

        # Check if too long
        if len(self.replay_memory) > self.params['memory_limit']:
            self.replay_memory.popleft()

        # Train
        if len(self.replay_memory) > self.params['training_start']:
            # Update Q-value function
            self.train()

            #if self.sympathetic_mode:
            #    self.robot_nextState_update()
            #    self.human_nextState_update()

        # Save model
        if (self.params['save_file']):
            if (self.episodesSoFar % self.params['save_interval'] == 0) and (done):
                layout_name = "GridWorld"

                # Save Sympathetic model 
                #name = layout_name + '_LearningAgent'
                #save_location = save_dir + name + "_" + str(self.episodesSoFar)
                #self.q_net.model.save(save_location)
                
                #if self.empathetic_mode:
                #    name = layout_name + '_GreedyRobot_' + str(int(not self.sympathetic_mode))
                #    save_location = save_dir + name + "_" + str(self.episodesSoFar)
                #    self.q_net_other['human'].Greedy_model.save(save_location)

                if self.bench_mark == 0:
                    if self.empathetic_mode:
                        name = layout_name + '_StateTransform_human'
                        save_location = save_dir + name + "_" + str(self.episodesSoFar)
                        self.q_net_other['human'].state_model.save(save_location)

                        name = layout_name + '_StateTransform_adversary'
                        save_location = save_dir + name + "_" + str(self.episodesSoFar)
                        self.q_net_other['adversary'].state_model.save(save_location)
                
    def save_other_step(self, obs, act, nextobs, done, reward_vector, door, door_ns, name):

        #if self.sympathetic_mode:
        experience = [obs, act, nextobs, done, reward_vector, door, door_ns]
        self.human_replay_memory[name].append(experience)

        if len(self.IRL_memory[name]) == 0:
            self.IRL_memory[name].append(experience)
        else:
            inside = False
            for a in range(len(self.IRL_memory[name])):
                if ((experience[0][0][1] == self.IRL_memory[name][a][0][0][1]).all()) and (experience[0][2] == self.IRL_memory[name][a][0][2]).all() and (experience[5] == self.IRL_memory[name][a][5]) and (experience[1] == self.IRL_memory[name][a][1]):
                    inside = True
                if inside == True:
                    break

            if inside == False:
                self.IRL_memory[name].append(experience)

        # Check if too long
        if len(self.human_replay_memory[name]) > self.params['memory_limit']:
            self.human_replay_memory[name].popleft()

        # Check if too long
        if len(self.IRL_memory[name]) > self.params['memory_limit']:
            self.IRL_memory[name].popleft()

        if len(self.human_replay_memory[name]) > self.params['training_start']:
            if (self.recreated_state_error < self.params['recreate_error']) and (self.start_train_other == False):
                print(self.episode_count)
                self.start_train_other = True

                # Save the recreate model
                file_name = 'GridWorld_ReCreate_State'
                save_location = save_dir + file_name
                self.stateRecreate.state_model.save(save_location)

                self.q_net_other['human'].state_model.set_weights(self.stateRecreate.state_model.get_weights())
                self.q_net_other['adversary'].state_model.set_weights(self.stateRecreate.state_model.get_weights())
                
            if self.start_train_other: 
                self.train_others_model(name)
            else:
                if len(self.replay_memory) > self.params['batch_multiple']*self.params['training_start']:
                    self.train_recreatestate()
                else:
                    pass

        if len(self.IRL_memory[name]) > self.params['training_start']:
            self.reward_prediction(name)
            self.greedy_robot_reward_prediction()
            self.robot_reward_prediction()

    def train(self):

        def QvalueFunction_setting(numerator_Qg, numerator_Qp, Q_differences):

            c = 1
            x = c*(numerator_Qg-numerator_Qp)/max(Q_differences)

            B1 = 1/(1+np.exp(-1*x))
            B2 = (1-B1)

            return B1, B2

        if (sum(self.other_rewards['human']) == 0) or (sum(self.other_rewards['adversary']) == 0):
            l1_scale_H = 1
            l1_scale_A = 1
        else:
            if self.empathetic_mode:
                l1_scale_H = 1
                l1_scale_A = 1
            else:
                l1_scale_H = np.sum(np.abs(self.robot_reward_vector))/np.sum(np.abs(self.other_rewards['human']))
                l1_scale_A = np.sum(np.abs(self.robot_reward_vector))/np.sum(np.abs(self.other_rewards['adversary']))

        batch = random.sample(self.replay_memory, self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_pos = extract(obs_full,3)
        a_pos_x = np.array(extract(a_pos,0)).reshape(-1,1)
        a_pos_y = np.array(extract(a_pos,1)).reshape(-1,1)

        actions = extract(batch,1)
        rewards = np.array(extract(batch,2))

        reward_human_vector = np.array(extract(batch,5))
        reward_human = np.array(extract(batch,5))
        reward_human = np.dot(reward_human,l1_scale_H*self.other_rewards['human'])

        reward_adversary_vector = np.array(extract(batch,5))
        reward_adversary = np.array(extract(batch,5))
        reward_adversary = np.dot(reward_adversary,l1_scale_A*self.other_rewards['adversary'])

        # Full information of state (s3)
        next_obs_fill = extract(batch,3)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_nextpos = extract(next_obs_fill,3)
        a_nextpos_x = np.array(extract(a_nextpos,0)).reshape(-1,1)
        a_nextpos_y = np.array(extract(a_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,10)

        door_s = np.array(extract(batch, 6)).reshape(-1,1)
        door_ns2 = np.array(extract(batch, 9)).reshape(-1,1)
        door_ns = np.array(extract(batch, 7)).reshape(-1,1)

        # Next possible actions
        next_actions = [self.possibleActions(r_nextpos[i]*[self.grid_width,self.grid_height], self.walls, door_ns[i], h_nextpos[i]*[self.grid_width,self.grid_height]) for i in range(self.params['batch_size'])]

        # Next state q-value
        q_vals = self.q_net.model.predict([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y])
        q_vals_next = self.q_net_target.model.predict([next_obs_r, door_ns, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y, a_nextpos_x, a_nextpos_y])
        q_vals_next = [np.max(q_vals_next[i][next_actions[i]]) for i in range(self.params['batch_size'])]#np.max(q_vals_next[next_actions],axis=1)

        # Greedy agent next values
        q_vals_greedy = self.q_net_greedy.model.predict([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y])
        q_vals_greedy_next = self.q_net_greedy_target.model.predict([next_obs_r, door_ns, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y, a_nextpos_x, a_nextpos_y])
        q_vals_greedy_next = [np.max(q_vals_greedy_next[i][next_actions[i]]) for i in range(self.params['batch_size'])]

        B1_val = []
        B2_val = []
        B3_val = []

        for i in range(self.params['batch_size']):

            denominator_differences = []

            # Beta value
            if (not self.sympathetic_mode):
                B1 = 1
                B2 = 0
                B3 = 0

            else:
                B1 = 0.75
                B2 = 1-B1
                B3 = 1-B1

            B1_val.append(B1)
            B2_val.append(B2)
            B3_val.append(B3)

            R_symp = B1*rewards[i] - B2*reward_human[i] - B3*reward_adversary[i]

            if terminal[i] == False:
                q_vals[i][actions[i]] = R_symp + self.params['discount']*q_vals_next[i]
                q_vals_greedy[i][actions[i]] = rewards[i] + self.params['discount']*q_vals_greedy_next[i]
            else:
                q_vals[i][actions[i]] = R_symp
                q_vals_greedy[i][actions[i]] = rewards[i]

            
            if reward_human_vector[i][0] == 1:
                self.beta_foodh += B1
                self.beta_foodh_cnt += 1
                self.reward_foodh += R_symp 
                self.reward_foodh_pnext += R_symp + self.params['discount']*q_vals_next[i]

            if reward_human_vector[i][1] == 1:
                self.beta_fooda += B1
                self.beta_fooda_cnt += 1
                self.reward_fooda += R_symp 
                self.reward_fooda_pnext += R_symp + self.params['discount']*q_vals_next[i]

            if reward_human_vector[i][2] == 1:
                self.beta_food += B1
                self.beta_food_cnt += 1
                self.reward_food += R_symp
                self.reward_food_pnext += R_symp + self.params['discount']*q_vals_next[i]
            
            #if reward_human_vector[i][2] == 1:
            #    self.beta_door_value += B1
            #    self.beta_door_cnt += 1
            #    self.reward_door += R_symp
            #    self.reward_door_pnext += R_symp + self.params['discount']*q_vals_next[i]

            #if reward_human_vector[i][3] == 1:
            #    self.beta_win += B1
            #    self.beta_win_cnt += 1
            #    self.reward_win += R_symp
            #    self.reward_win_pnext += R_symp + self.params['discount']*q_vals_next[i]

            if reward_human_vector[i][3] == 1:
                self.beta_step += B1
                self.beta_step_cnt += 1
                self.reward_step += R_symp
                self.reward_step_pnext += R_symp + self.params['discount']*q_vals_next[i]


        self.q_net.model.fit([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y], q_vals, shuffle=True, epochs = 5, batch_size = self.params['batch_size'], verbose = 0)

        self.q_net_greedy.model.fit([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y], q_vals_greedy, shuffle=True, epochs = 5, batch_size = self.params['batch_size'], verbose = 0)

        error = self.q_net.model.history.history['loss'][-1]
        error_greedy = self.q_net_greedy.model.history.history['loss'][-1]
        self.greedy_cost_total += error_greedy

        self.cost_total += error
        self.cnt += 1

    def prepare_other_batch(self, batch_size, name):

        batch = random.sample(self.human_replay_memory[name], batch_size)

        obs_full = extract(batch,0)

        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(batch_size,self.params['view_window_width'],self.params['view_window_height'],1)
        
        if name == 'human':
            obs_h = extract(obs,1)
            obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
            obs_h = rgb2gray(obs_h).reshape(batch_size,self.params['view_window_width'],self.params['view_window_height'],1)
        elif name == 'adversary':
            obs_h = extract(obs,2)
            obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
            obs_h = rgb2gray(obs_h).reshape(batch_size,self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)


        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_pos = extract(obs_full,3)
        a_pos_x = np.array(extract(a_pos,0)).reshape(-1,1)
        a_pos_y = np.array(extract(a_pos,1)).reshape(-1,1)

        actions = extract(batch,1)

        # Full information of state
        next_obs_full = extract(batch,2)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_full,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(batch_size,self.params['view_window_width'],self.params['view_window_height'],1)

        if name == 'human':
            next_obs_h = extract(next_obs,1)
            next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
            next_obs_h = rgb2gray(next_obs_h).reshape(batch_size,self.params['view_window_width'],self.params['view_window_height'],1)
        elif name == 'adversary':
            next_obs_h = extract(next_obs,2)
            next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
            next_obs_h = rgb2gray(next_obs_h).reshape(batch_size,self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_full,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_full,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        a_nextpos = extract(next_obs_full,3)
        a_nextpos_x = np.array(extract(a_nextpos,0)).reshape(-1,1)
        a_nextpos_y = np.array(extract(a_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,3)

        door_s = np.array(extract(batch,5)).reshape(-1,1)
        door_ns = np.array(extract(batch,6)).reshape(-1,1)

        return obs_r, obs_h, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y, actions, next_obs_h, next_obs_r, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y, a_nextpos_x, a_nextpos_y, terminal, door_s, door_ns

    def train_recreatestate(self):

        #Training self.stateRecreate
        # Prepare a batch
        batch = random.sample(self.replay_memory, self.params['batch_multiple']*self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_multiple']*self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        hist = self.stateRecreate.state_model.fit(obs_r, obs_r, shuffle=True, epochs=50, batch_size=self.params['batch_multiple']*self.params['batch_size'],verbose=0)
        self.recreated_state_error = np.mean(hist.history['loss'])

    def train_others_model(self, name):

        def TD_error():

            # Validation batch
            obs_r, obs_h, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y, actions, next_obs_h, next_obs_r, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y, a_nextpos_x, a_nextpos_y, terminal, door_s, door_ns = self.prepare_other_batch(30, name)

            inputqg = [obs_h, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y]
            inputqg = rgb_transform(inputqg, name)

            # Calculate TD error:
            q_vals = np.array(self.q_net_other[name].model.predict(inputqg))
            perc_correct = np.sum(np.argmax(q_vals,1) == actions)/len(actions)

            return perc_correct

        # Prepare a batch
        obs_r, obs_h, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y, actions, next_obs_h, next_obs_r, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y, a_nextpos_x, a_nextpos_y, terminal, door_s, door_ns = self.prepare_other_batch(self.params['batch_size'], name)

        # Convert actions to onehot encoding
        q_vals = np.zeros((self.params['batch_size'],self.params['num_human_actions']))

        inputqg = [obs_h, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y]
        inputqg = rgb_transform(inputqg, name)

        pred_q_vals = np.array(self.q_net_other[name].model.predict(inputqg))

        for i in range(self.params['batch_size']):
            for r in range(self.params['num_human_actions']):
                if (np.argmax(pred_q_vals[i]) == actions[i]) and (r==actions[i]):
                    q_vals[i][r] = 1 #pred_q_vals[i][r]
                elif (r == actions[i]):
                    q_vals[i][actions[i]] = 1
                else:
                    q_vals[i][r] = 0 #pred_q_vals[i][r]
        
        inputqg = [obs_h, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, a_pos_x, a_pos_y]
        inputqg = rgb_transform(inputqg, name)
        hist = self.q_net_other[name].model.fit(inputqg, q_vals, shuffle=True, epochs=10, batch_size=self.params['batch_size'],verbose=0)

        if self.empathetic_mode:
            # Transfer over weights of learning agent
            self.q_net_other[name].Greedy_model.set_weights(self.q_net_greedy_target.model.get_weights())

        perc_error = TD_error()

        self.human_cost_total += np.mean(hist.history['loss'])
        self.human_perc_correct += perc_error
        self.cnt_human += 1

    def reward_prediction(self, name):

        batch = random.sample(self.IRL_memory[name], self.params['batch_size'])

        obs_full = extract(batch,0)

        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        
        if name == 'human':
            obs_h = extract(obs,1)
            obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
            obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        elif name == 'adversary':
            obs_h = extract(obs,2)
            obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
            obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_pos = extract(obs_full,3)
        a_pos_x = np.array(extract(a_pos,0)).reshape(-1,1)
        a_pos_y = np.array(extract(a_pos,1)).reshape(-1,1)

        actions = extract(batch,1)

        # Full information of state
        next_obs_full = extract(batch,2)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_full,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        if name == 'human':
            next_obs_h = extract(next_obs,1)
            next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
            next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        elif name == 'adversary':
            next_obs_h = extract(next_obs,2)
            next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
            next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_full,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_full,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_nextpos = extract(next_obs_full,3)
        a_nextpos_x = np.array(extract(a_nextpos,0)).reshape(-1,1)
        a_nextpos_y = np.array(extract(a_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,3)

        reward_vector = extract(batch,4)

        door_s = np.array(extract(batch,5)).reshape(-1,1)
        door_ns = np.array(extract(batch,6)).reshape(-1,1)

        R_target = []
        rewardfeaturelist = []

        for i in range(self.params['batch_size']):

            inputqg = [obs_h[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_s[i].reshape(1,1), r_pos_x[i].reshape(1,1), r_pos_y[i].reshape(1,1), h_pos_x[i].reshape(1,1), h_pos_y[i].reshape(1,1), a_pos_x[i].reshape(1,1), a_pos_y[i].reshape(1,1)]
            inputqg = rgb_transform(inputqg, name)

            q = self.q_net_other[name].model_bsm.predict(inputqg)
            q = float(q[0][actions[i]])

            if terminal[i] == True:
                q_max = 0.0
            else:

                inputqg = [next_obs_h[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_ns[i].reshape(1,1), r_nextpos_x[i].reshape(1,1), r_nextpos_y[i].reshape(1,1), h_nextpos_x[i].reshape(1,1), h_nextpos_y[i].reshape(1,1), a_nextpos_x[i].reshape(1,1), a_nextpos_y[i].reshape(1,1)]
                inputqg = rgb_transform(inputqg, name)

                q_max = float(max(self.q_net_other[name].model_bsm.predict(inputqg)[0]))

            R = q - self.params['discount']*q_max

            R_target.append(R)
            rewardfeaturelist.append(reward_vector[i])

            pred_R = np.dot(self.other_rewards[name], reward_vector[i])

            difference_R = (pred_R - R)
            self.other_rewards[name] = self.other_rewards[name] - (0.02 * reward_vector[i] * difference_R)

        # Rough estimate of reward error
        error = 0
        for i in range(self.params['batch_size']):
            error +=  np.sum(np.abs(np.dot(self.other_rewards[name], rewardfeaturelist[i]) - R_target[i]))

        self.IRL_error += error

    def greedy_robot_reward_prediction(self):

        batch = random.sample(self.replay_memory, self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_pos = extract(obs_full,3)
        a_pos_x = np.array(extract(a_pos,0)).reshape(-1,1)
        a_pos_y = np.array(extract(a_pos,1)).reshape(-1,1)

        actions = extract(batch,1)

        # Full information of state (s3)
        next_obs_fill = extract(batch,3)

        reward_vector = np.array(extract(batch,5))

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_nextpos = extract(next_obs_fill,3)
        a_nextpos_x = np.array(extract(a_nextpos,0)).reshape(-1,1)
        a_nextpos_y = np.array(extract(a_nextpos,1)).reshape(-1,1)

        door_ns2 = np.array(extract(batch, 9)).reshape(-1,1)
        ###################################

        terminal = extract(batch,10)

        door_s = np.array(extract(batch, 6)).reshape(-1,1)
        door_ns = np.array(extract(batch, 7)).reshape(-1,1)

        # Greedy Agent reward check
        R_target = []
        rewardfeaturelist = []
        for i in range(self.params['batch_size']):

            q = self.q_net_greedy.model.predict([obs_r[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_s[i].reshape(1,1), r_pos_x[i].reshape(1,1), r_pos_y[i].reshape(1,1), h_pos_x[i].reshape(1,1), h_pos_y[i].reshape(1,1), a_pos_x[i].reshape(1,1), a_pos_y[i].reshape(1,1)])
            q = float(q[0][actions[i]])

            if terminal[i] == True:
                q_max = 0.0
            else:
                q_max = float(max(self.q_net_greedy.model.predict([next_obs_r[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_ns[i].reshape(1,1), r_nextpos_x[i].reshape(1,1), r_nextpos_y[i].reshape(1,1), h_nextpos_x[i].reshape(1,1), h_nextpos_y[i].reshape(1,1), a_nextpos_x[i].reshape(1,1), a_nextpos_y[i].reshape(1,1)])[0]))

            R = q - self.params['discount']*q_max

            R_target.append(R)
            rewardfeaturelist.append(reward_vector[i])

            pred_R = np.dot(self.greedy_rewards, reward_vector[i])

            difference_R = (pred_R - R)
            self.greedy_rewards = self.greedy_rewards - (0.2 * reward_vector[i] * difference_R)

    def robot_reward_prediction(self):

        batch = random.sample(self.replay_memory, self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        a_pos = extract(obs_full,3)
        a_pos_x = np.array(extract(a_pos,0)).reshape(-1,1)
        a_pos_y = np.array(extract(a_pos,1)).reshape(-1,1)

        actions = extract(batch,1)

        # Full information of state (s3)
        next_obs_fill = extract(batch,3)

        reward_vector = np.array(extract(batch,5))

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        # Extract x and y position of adversary
        a_nextpos = extract(next_obs_fill,3)
        a_nextpos_x = np.array(extract(a_nextpos,0)).reshape(-1,1)
        a_nextpos_y = np.array(extract(a_nextpos,1)).reshape(-1,1)

        door_ns2 = np.array(extract(batch, 9)).reshape(-1,1)
        ###################################

        terminal = extract(batch,10)

        door_s = np.array(extract(batch, 6)).reshape(-1,1)
        door_ns = np.array(extract(batch, 7)).reshape(-1,1)

        # Greedy Agent reward check
        R_target = []
        rewardfeaturelist = []
        for i in range(self.params['batch_size']):

            q = self.q_net.model.predict([obs_r[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_s[i].reshape(1,1), r_pos_x[i].reshape(1,1), r_pos_y[i].reshape(1,1), h_pos_x[i].reshape(1,1), h_pos_y[i].reshape(1,1), a_pos_x[i].reshape(1,1), a_pos_y[i].reshape(1,1)])
            q = float(q[0][actions[i]])

            if terminal[i] == True:
                q_max = 0.0
            else:
                q_max = float(max(self.q_net.model.predict([next_obs_r[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_ns[i].reshape(1,1), r_nextpos_x[i].reshape(1,1), r_nextpos_y[i].reshape(1,1), h_nextpos_x[i].reshape(1,1), h_nextpos_y[i].reshape(1,1), a_nextpos_x[i].reshape(1,1), a_nextpos_y[i].reshape(1,1)])[0]))

            R = q - self.params['discount']*q_max

            R_target.append(R)
            rewardfeaturelist.append(reward_vector[i])

            pred_R = np.dot(self.robot_rewards_pred, reward_vector[i])

            difference_R = (pred_R - R)
            self.robot_rewards_pred = self.robot_rewards_pred - (0.2 * reward_vector[i] * difference_R)

    def reward_features(self, foodbefore_0, foodbefore_1, foodbefore_2, foodafter_0, foodafter_1, foodafter_2, door_status, last_door):
        # human_pellet, adversary_pellet, robot_pellet, opening door, finishing game, step

        reward_vector = np.array([0, 0, 0, 0])

        # Check if human has eaten pellet
        if len(foodafter_1) < len(foodbefore_1):
            reward_vector[0] = int(len(foodbefore_1)-len(foodafter_1))

        # Check if human has eaten pellet
        if len(foodafter_2) < len(foodbefore_2):
            reward_vector[1] = int(len(foodbefore_2)-len(foodafter_2))

        # Check if robot has eaten pellet
        if len(foodafter_0) < len(foodbefore_0):
            reward_vector[2] = int(len(foodbefore_0)-len(foodafter_0))

        # Check if door has been opened for the first time
        #if door_status:
        #    if last_door == False:
        #        reward_vector[3] = 1

        # Check if game is finished
        #if (len(foodafter_0) == 0) and (len(foodafter_1) == 0) and (len(foodafter_2) == 0):
        #    reward_vector[3] = 1

        # If none of the above occurs, then just taken a step
        if sum(reward_vector) == 0:
            reward_vector[3] = 1

        return reward_vector

    def start_episode(self):
        self.cumulative_reward = 0
        self.step_count = 0

    def end_episode(self):
        #print(
        #    f"Finished episode {self.episode_count} after {self.step_count} steps."
        #    f"  Episode return was {self.cumulative_reward}."
        #    f"  Epsilon was {self.epsilon}."
        #)
        self.episode_count += 1

# Other useful functions
def extract(lst, pos):
    return [item[pos] for item in lst]

def rgb2gray(rgb):

    # 5 decimal rounding of values
    robot_food_value = 0.12876 
    human_food_value = 0.38162
    adversary_food_value = 0.12814
    button_value = 0.25286
    human_value = 1.15848 
    robot_value = 0.39087
    adversary_value = 0.40154
    gate_value = 1.02951
    other_agent_value = 0.85

    rgb_converted = np.dot(rgb, [0.2989, 0.5870, 0.1140])

    # Recode adversary values
    #other = np.where(np.round(rgb_converted,5) == adversary_value)
    #if len(other[0]) != 0:
    #    if len(rgb_converted.shape) == 3:
    #        rgb_converted[other[0][:],other[1][:],other[2][:]] = 0.5 # Adversary
    #    else:
    #        rgb_converted[other[0][:],other[1][:]] = 0.5 # Adversary

    # Clear out center cell
    if len(rgb_converted.shape) == 3:
        rgb_converted[:,int(rgb_converted.shape[1]/2), int(rgb_converted.shape[2]/2)] = 0 
    else:
        rgb_converted[int(rgb_converted.shape[0]/2), int(rgb_converted.shape[1]/2)] = 0 
    

    # Recode value of other agent
    other = np.where(np.round(rgb_converted,5) == robot_value)
    if len(other[0]) != 0:
        if len(rgb_converted.shape) == 3:
            rgb_converted[other[0][:],other[1][:],other[2][:]] = other_agent_value # Robot
        else:
            rgb_converted[other[0][:],other[1][:]] = other_agent_value # Robot

    other = np.where(np.round(rgb_converted,5) == human_value)
    if len(other[0]) != 0:
        if len(rgb_converted.shape) == 3:
            rgb_converted[other[0][:],other[1][:],other[2][:]] = other_agent_value # Human
        else:
            rgb_converted[other[0][:],other[1][:]] = other_agent_value # Human

    other = np.where(np.round(rgb_converted,5) == adversary_value)
    if len(other[0]) != 0:
        if len(rgb_converted.shape) == 3:
            rgb_converted[other[0][:],other[1][:],other[2][:]] = other_agent_value # Human
        else:
            rgb_converted[other[0][:],other[1][:]] = other_agent_value # Human

    # Recode adversary food
    other = np.where(np.round(rgb_converted,5) == adversary_food_value)
    if len(other[0]) != 0:
        if len(rgb_converted.shape) == 3:
            rgb_converted[other[0][:],other[1][:],other[2][:]] = 0.5 # Adversary
        else:
            rgb_converted[other[0][:],other[1][:]] = 0.5 # Adversary

    # Recode gate_value 
    #other = np.where(np.round(rgb_converted,5) == gate_value)
    #if len(other[0]) != 0:
    #    if len(rgb_converted.shape) == 3:
    #        rgb_converted[other[0][:],other[1][:],other[2][:]] = 0.65 # Robot
    #    else:
    #        rgb_converted[other[0][:],other[1][:]] = 0.65 # Robot
    
    # Recode robot food value of other agent
    #other = np.where(np.round(rgb_converted,5) == 0.12444)
    #if len(other[0]) != 0:
    #    if len(rgb_converted.shape) == 3:
    #        rgb_converted[other[0][:],other[1][:],other[2][:]] = 0.65 # Robot
    #    else:
    #        rgb_converted[other[0][:],other[1][:]] = 0.65 # Robot
    
    return rgb_converted

if bench_mark == 0:
    def rgb_transform(rgb, name):

        transformed_output = rgb

        return transformed_output
elif bench_mark == 1:
    def rgb_transform(rgb, name):

        # 5 decimal rounding of values
        robot_food_value = 0.12876 
        human_food_value = 0.38162
        adversary_food_value = 0.5
        button_value = 0.5 
        human_value = 1.15848 
        robot_value = 0.39087 
        gate_value = 1.02951
        other_agent_value = 0.85

        rgb_converted = rgb[0]
        button_status = (1-rgb[1])

        for i in range(rgb_converted.shape[0]):

            # Food of learning agent
            other_rf = np.where(np.round(rgb_converted[i,:,:,0],5) == robot_food_value)
            # Food of indep agent
            other_hf = np.where(np.round(rgb_converted[i,:,:,0],5) == human_food_value)
            # Food of other indep agent
            other_af = np.where(np.round(rgb_converted[i,:,:,0],5) == adversary_food_value)
            
            
            if len(other_rf[0]) != 0:
                if len(rgb_converted.shape) == 3:
                    rgb_converted[i,other_rf[0][:],other_rf[1][:],other_rf[2][:]] = human_food_value # Robot
                else:
                    rgb_converted[i,other_rf[0][:],other_rf[1][:],0] = human_food_value # Robot

            if name == 'human':
                if len(other_hf[0]) != 0:
                    if len(rgb_converted.shape) == 3:
                        rgb_converted[i,other_hf[0][:],other_hf[1][:],other_hf[2][:]] = robot_food_value # Robot
                    else:
                        rgb_converted[i,other_hf[0][:],other_hf[1][:],0] = robot_food_value # Robot
            elif name == 'adversary':
                if len(other_af[0]) != 0:
                    if len(rgb_converted.shape) == 3:
                        rgb_converted[i,other_af[0][:],other_af[1][:],other_af[2][:]] = robot_food_value # Robot
                    else:
                        rgb_converted[i,other_af[0][:],other_af[1][:],0] = robot_food_value # Robot

        transformed_output = [rgb_converted, button_status, rgb[2], rgb[3], rgb[4], rgb[5], rgb[6], rgb[7]]

        return transformed_output
elif bench_mark == 2:
    def rgb_transform(rgb, name):

        # 5 decimal rounding of values
        robot_food_value = 0.12876 
        human_food_value = 0.38162
        adversary_food_value = 0.5
        button_value = 0.5 
        human_value = 1.15848 
        robot_value = 0.39087 
        gate_value = 1.02951
        other_agent_value = 0.85

        rgb_converted = rgb[0]
        button_status = (1-rgb[1])

        for i in range(rgb_converted.shape[0]):

            # Food of learning agent
            other_rf = np.where(np.round(rgb_converted[i,:,:,0],5) == robot_food_value)
            # Food of indep agent
            other_hf = np.where(np.round(rgb_converted[i,:,:,0],5) == human_food_value)
            # Food of other indep agent
            other_af = np.where(np.round(rgb_converted[i,:,:,0],5) == adversary_food_value)
            
            
            if len(other_rf[0]) != 0:
                if len(rgb_converted.shape) == 3:
                    rgb_converted[i,other_rf[0][:],other_rf[1][:],other_rf[2][:]] = adversary_food_value # Robot
                else:
                    rgb_converted[i,other_rf[0][:],other_rf[1][:],0] = adversary_food_value # Robot

            if name == 'human':
                if len(other_hf[0]) != 0:
                    if len(rgb_converted.shape) == 3:
                        rgb_converted[i,other_hf[0][:],other_hf[1][:],other_hf[2][:]] = robot_food_value # Robot
                    else:
                        rgb_converted[i,other_hf[0][:],other_hf[1][:],0] = robot_food_value # Robot

            elif name == 'adversary':
                if len(other_af[0]) != 0:
                    if len(rgb_converted.shape) == 3:
                        rgb_converted[i,other_af[0][:],other_af[1][:],other_af[2][:]] = robot_food_value # Robot
                    else:
                        rgb_converted[i,other_af[0][:],other_af[1][:],0] = robot_food_value # Robot

        transformed_output = [rgb_converted, button_status, rgb[2], rgb[3], rgb[4], rgb[5], rgb[6], rgb[7]]

        return transformed_output

def rgb2gray_old(rgb):
    rgb_converted = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    rgb_converted = 2*rgb_converted
    rgb_converted[rgb_converted > 1] = 1
    return rgb_converted

# Create the environment based on the combined env/player config
env = gym.make('MarlGrid-3AgentSympathy15x15-v0')

# Specify run type
empathetic_mode = empathy_mode == "true"
sympathetic_mode = CSI_mode == "true"
Ghost_model_type = model_type # or 'state'
loss1_weight = gamma
training_mode = training_mode == "true"

# to run greedy: empathetic_mode = False, sympathetic_mode = False
# to run sympathetic: empathetic_mode = False, sympathetic_mode = True
# to run empathetic: empathetic_mode = True, sympathetic_mode = True

# Create a human player interface per the class defined above
agents = TestRLAgent(sympathetic_mode, empathetic_mode, Ghost_model_type, loss1_weight = loss1_weight, bench_mark = bench_mark, training_mode = training_mode)

print([empathetic_mode, sympathetic_mode, Ghost_model_type, loss1_weight, log_dir, save_dir, bench_mark, training_mode])

if training_mode:

    loops = range(1)
    episode_range = range(agents.params['no_episodes'])
else:

    loops = np.arange(0,agents.params['no_episodes'],agents.params['save_interval'])
    episode_range = range(agents.params['no_tests'])

for loop in loops:

    if training_mode:
        # Create a human player interface per the class defined above
        agents = TestRLAgent(sympathetic_mode, empathetic_mode, Ghost_model_type, loss1_weight = loss1_weight, bench_mark = bench_mark, training_mode = training_mode)
    else:
        # Create a human player interface per the class defined above
        agents = TestRLAgent(sympathetic_mode, empathetic_mode, Ghost_model_type, loss1_weight = loss1_weight, bench_mark = bench_mark, training_mode = training_mode, load_file = str(loop))

    agents.grid_width = env.grid.width
    agents.grid_height = env.grid.height

    for episodes in episode_range:

        print("Episode:", str(episodes))

        # Start an episode!
        # Each observation from the environment contains a list of observaitons for each agent.
        # In this case there's only one agent so the list will be of length one.
        obs_list = env.reset()

        agents.walls = env.walls.copy()

        agents.start_episode()
        done = False
        won = False
        human_done = False
        adversary_done = False
        stop_human_training = False
        stop_adversary_training = False

        agents.episodesSoFar = episodes

        lastR_state = None
        lastR_door = None
        lastH_state = None
        lastH_door = None
        lastA_state = None
        lastA_door = None

        R_human_food = 0
        R_adversary_food = 0

        robot_action = None
        human_action = None
        lastR_food = None
        lastH_food = None

        door_was_opened = False

        food_obs1 = [None, None]
        food_obs2 = [None, None]
        food_obs3 = [None, None]

        while not done:

            for agent_no in [0,1,2]:

                #if door_was_opened:
                #env.render() # OPTIONAL: render the whole scene + birds eye view
                #time.sleep(0.1)
                if agents.training_mode:
                    agents.epsilon = max(0.1,agents.params['eps_initial']*(agents.params['eps_decay']**episodes))
                else:
                    agents.epsilon = 0
                    agents.epsilon_model_switch = 0

                # Get location of all available food (human)
                humanfood_pos = env.agents[1].foodpos
                adversaryfood_pos = env.agents[2].foodpos


                # Check whether door is open
                door_status = env.agents[0].door_opened

                if door_status:
                    door_was_opened = True

                if (agent_no == 0) and (lastH_state is not None) and (lastA_state is not None):
                    # Save state, action, reward and next state for robot
                    reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)
                    #if env.agents[0].door_opened:
                    #    if lastR_door:
                    #        reward_vector[2] = 0
                    reward_vector[3] = 1

                    # reward calculation
                    # human_food, robot_food, opening door, finishing game, step
                    robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                    agents.cumulative_reward += robot_reward
                    agents.step_count += 1

                    if agents.training_mode:

                        reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)

                        current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                        agents.save_step(
                            lastR_state, robot_action, robot_reward, lastH_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), lastH_door, won
                        )

                if agents.training_mode:
                    if not stop_human_training:
                        if (agent_no == 1) and (lastH_state is not None):
                            # Save state, action, reward and next state for robot
                            reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], lastH_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastH_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            agents.save_other_step(
                                lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened), 'human'
                            )

                            if human_done:
                                stop_human_training = True

                    if not stop_adversary_training:
                        if (agent_no == 2) and (lastA_state is not None):
                            # Save state, action, reward and next state for robot
                            reward_vector = agents.reward_features(lastA_food[0], lastA_food[1], lastA_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastA_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            agents.save_other_step(
                                lastA_state, adversary_action, current_state, adversary_done, reward_vector, int(lastA_door), int(env.agents[0].door_opened),'adversary'
                            )

                            if adversary_done:
                                stop_adversary_training = True

                # Output action for each agent and whether robot has picked up key
                action_array = agents.action_step(obs_list[0], env.agents[0].pos, env.agents[1].pos, env.agents[2].pos, humanfood_pos, adversaryfood_pos, agents.walls, door_status,agent_no)

                if agent_no == 0:

                    lastR_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]
                    lastR_door = door_status
                    robot_action = action_array[0]
                    lastR_food = [env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos]

                elif agent_no == 1:

                    lastH_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]
                    lastH_door = door_status
                    human_action = action_array[1]
                    lastH_food = [env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos]

                elif agent_no == 2:

                    lastA_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]
                    lastA_door = door_status
                    adversary_action = action_array[2]
                    lastA_food = [env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos]

                # Simulate next action (code adapted to moving in the direction indicated.)
                # action = 0 (east), action = 1 (south), action = 2 (west), action = 3 (north)
                if action_array[agent_no] == 4:
                    if agent_no == 0:
                        action_array_tmp = (6,action_array[1],action_array[2])

                    elif agent_no == 1:
                        action_array_tmp = (action_array[0],6,action_array[2])
                    
                    elif agent_no == 2:
                        action_array_tmp = (action_array[0],action_array[1],6)

                    next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                elif action_array[agent_no] == env.agents[agent_no].dir:
                    if agent_no == 0:
                        action_array_tmp = (2,action_array[1],action_array[2])
                    elif agent_no == 1:
                        action_array_tmp = (action_array[0],2,action_array[2])
                    elif agent_no == 2:
                        action_array_tmp = (action_array[0],action_array[1],2)
                    next_obs_list, rew_list, done, _ = env.step(action_array_tmp)
                else:
                    while action_array[agent_no] != env.agents[agent_no].dir:
                        if agent_no == 0:
                            action_array_tmp = (0,action_array[1],action_array[2])
                        elif agent_no == 1:
                            action_array_tmp = (action_array[0],0,action_array[2])
                        elif agent_no == 2:
                            action_array_tmp = (action_array[0],action_array[1],0)
                        next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                    if action_array[agent_no] == env.agents[agent_no].dir:
                        if agent_no == 0:
                            action_array_tmp = (2,action_array[1],action_array[2])
                        elif agent_no == 1:
                            action_array_tmp = (action_array[0],2,action_array[2])
                        elif agent_no == 2:
                            action_array_tmp = (action_array[0],action_array[1],2)
                        next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                """
                # Opening the door if in the position of the button
                # Check if stepped on button while door is closed
                if agent_no == 0:
                    if ((list(env.agents[0].pos) == [7,3]) and (not door_status)):
                        #env.render()
                        if env.agents[agent_no].dir == 3:
                            action_array_tmp = (3,action_array[1],action_array[2])
                        else:
                            while env.agents[agent_no].dir != 3:
                                action_array_tmp = (0,action_array[1],action_array[2])
                                next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                            action_array_tmp = (3,action_array[1],action_array[2])
                            next_obs_list, rew_list, done, _ = env.step(action_array_tmp)
                        
                        next_obs_list, rew_list, done, _ = env.step(action_array_tmp)
                """

                    #env.render()

                # Check how many of human and adversary pellets consumed
                if (len(env.agents[1].foodpos) < len(lastR_food[1])) and (agent_no == 0):
                    R_human_food += 1

                if (len(env.agents[2].foodpos) < len(lastR_food[2])) and (agent_no == 0):
                    R_adversary_food += 1

                # If robot has finished all of its own food, game will end
                if (len(env.agents[0].foodpos) == 0) and (len(env.agents[1].foodpos)==0) and (len(env.agents[2].foodpos)==0):
                    done = True
                    won = True

                if len(env.agents[1].foodpos) == 0:
                    human_done = True

                if len(env.agents[2].foodpos) == 0:
                    adversary_done = True

                obs_list = next_obs_list.copy()

                # Check if game ended
                if done:
                    if (agent_no == 0):
                        # Save state, action, reward and next state for robot
                        reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)
                        #if env.agents[0].door_opened:
                        #    if lastR_door:
                        #        reward_vector[2] = 0
                        reward_vector[3] = 1

                        # reward calculation
                        # human_food, robot_food, opening door, finishing game, step
                        robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                        agents.cumulative_reward += robot_reward
                        agents.step_count += 1

                        if agents.training_mode:
                            reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]),env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            agents.save_step(
                                lastR_state, robot_action, robot_reward, current_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), int(env.agents[0].door_opened), won
                            )

                            # Save state, action, reward and next state for robot
                            reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], lastH_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastH_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]
                            if not stop_human_training:
                                agents.save_other_step(
                                    lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened), 'human'
                                )

                            # Save state, action, reward and next state for adversary
                            reward_vector = agents.reward_features(lastA_food[0], lastA_food[1], lastA_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastA_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]
                            if not stop_adversary_training:
                                agents.save_other_step(
                                    lastA_state, adversary_action, current_state, adversary_done, reward_vector, int(lastA_door), int(env.agents[0].door_opened), 'adversary'
                                )

                    elif (agent_no == 1):

                        # Save state, action, reward and next state for robot
                        reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)
                        #if env.agents[0].door_opened:
                        #    if lastR_door:
                        #        reward_vector[2] = 0
                        reward_vector[3] = 1

                        # reward calculation
                        # human_food, robot_food, opening door, finishing game, step
                        robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                        agents.cumulative_reward += robot_reward
                        agents.step_count += 1

                        if agents.training_mode:
                            reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            agents.save_step(
                                lastR_state, robot_action, robot_reward, lastH_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), lastH_door, won
                            )

                            # Save state, action, reward and next state for robot
                            reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], lastH_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastH_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            if not stop_human_training:
                                agents.save_other_step(
                                    lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened),'human'
                                )

                            # Save state, action, reward and next state for adversary
                            reward_vector = agents.reward_features(lastA_food[0], lastA_food[1], lastA_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastA_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            if not stop_adversary_training:
                                agents.save_other_step(
                                    lastA_state, adversary_action, current_state, adversary_done, reward_vector, int(lastA_door), int(env.agents[0].door_opened),'adversary'
                                )

                    elif (agent_no == 2):

                        # Save state, action, reward and next state for robot
                        reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)
                        #if env.agents[0].door_opened:
                        #    if lastR_door:
                        #        reward_vector[2] = 0
                        reward_vector[3] = 1

                        # reward calculation
                        # human_food, robot_food, opening door, finishing game, step
                        robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                        agents.cumulative_reward += robot_reward
                        agents.step_count += 1

                        if agents.training_mode:
                            reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], lastR_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastR_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            agents.save_step(
                                lastR_state, robot_action, robot_reward, lastH_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), lastH_door, won
                            )

                            # Save state, action, reward and next state for robot
                            reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], lastH_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastH_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            if not stop_human_training:
                                agents.save_other_step(
                                    lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened),'human'
                                )

                            # Save state, action, reward and next state for adversary
                            reward_vector = agents.reward_features(lastA_food[0], lastA_food[1], lastA_food[2], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[2].foodpos, env.agents[0].door_opened, lastA_door)

                            current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[2].pos/np.array([agents.grid_width, agents.grid_height])]

                            if not stop_adversary_training:
                                agents.save_other_step(
                                    lastA_state, adversary_action, current_state, adversary_done, reward_vector, int(lastA_door), int(env.agents[0].door_opened),'adversary'
                                )

                    break

        agents.end_episode()

        # Update Q-value target
        if agents.training_mode:
            if episodes % agents.params['target_update'] == 0:
                agents.q_net_target.model.set_weights(agents.q_net.model.get_weights())
                agents.q_net_greedy_target.model.set_weights(agents.q_net_greedy.model.get_weights())

        # Print out episode metrics
        if agents.cnt_human == 0:
            agents.cnt_human = 1
        if agents.cnt == 0:
            agents.cnt = 1
        if agents.beta_door_cnt == 0:
            agents.beta_door_cnt = 1
        if agents.beta_foodh_cnt == 0:
            agents.beta_foodh_cnt = 1
        if agents.beta_fooda_cnt == 0:
            agents.beta_fooda_cnt = 1
        if agents.beta_food_cnt == 0:
            agents.beta_food_cnt = 1
        if agents.beta_win_cnt == 0:
            agents.beta_win_cnt = 1
        if agents.beta_step_cnt == 0:
            agents.beta_step_cnt = 1

        if agents.training_mode:
            log_file = open(log_dir+'GreedyAgent' + str(int(not agents.sympathetic_mode)) +'.log','a')
            log_file.write("# %4d | steps_t: %5d |r: %5d | cost: %6f | cost_greedy: %6f | cost_human: %6f | human_perc: %6f | e: %10f | robot_foodleft: %2d | H_foodleft: %2d | R_consumedH: %2d | R_consumedA: %2d | won: %r \n" %
                                    (episodes, agents.cnt, agents.cumulative_reward, float(agents.cost_total)/float(agents.cnt), float(agents.greedy_cost_total)/float(agents.cnt), float(agents.human_cost_total)/float(agents.cnt_human), float(agents.human_perc_correct)/float(agents.cnt_human) , agents.epsilon, len(env.agents[0].foodpos), len(env.agents[1].foodpos), R_human_food, R_adversary_food, won))
        else:
            log_file = open(log_dir+'GreedyAgent' + str(int(not agents.sympathetic_mode)) + '_' + str(loop) +'_Test.log','a')
            log_file.write("# %4d | steps_t: %5d |r: %5d | cost: %6f | cost_greedy: %6f | cost_human: %6f | human_perc: %6f | e: %10f | robot_foodleft: %2d | H_foodleft: %2d | R_consumedH: %2d | R_consumedA: %2d | won: %r \n" %
                                        (episodes, agents.cnt, agents.cumulative_reward, float(agents.cost_total)/float(agents.cnt), float(agents.greedy_cost_total)/float(agents.cnt), float(agents.human_cost_total)/float(agents.cnt_human), float(agents.human_perc_correct)/float(agents.cnt_human) , agents.epsilon, len(env.agents[0].foodpos), len(env.agents[1].foodpos), R_human_food, R_adversary_food, won))

        if agents.training_mode:
            # Print out episode metrics
            #if agents.sympathetic_mode:
            log_file = open(log_dir+'IRL_rewards_IA1'+'.log','a')
            log_file.write("# %4d | reward: %r \n" %
                                        (episodes,list(agents.other_rewards['human'])))

            log_file = open(log_dir+'IRL_rewards_IA2'+'.log','a')
            log_file.write("# %4d | reward: %r \n" %
                                        (episodes,list(agents.other_rewards['adversary'])))

            # Print out episode metrics
            #if agents.sympathetic_mode:
            log_file = open(log_dir+'Greedy_IRL_rewards'+'.log','a')
            log_file.write("# %4d | reward: %r \n" %
                                        (episodes, list(agents.greedy_rewards)))

            #if agents.sympathetic_mode:
            log_file = open(log_dir+'Robot_IRL_rewards'+'.log','a')
            log_file.write("# %4d | reward: %r \n" %
                                        (episodes,list(agents.robot_rewards_pred)))
                                            
            # Print out beta at door opening
            if agents.sympathetic_mode:
                log_file = open(log_dir+'beta_door'+'.log','a')
                log_file.write("# %4d | beta: %r | beta_foodh: %r | beta_food: %r | beta_win: %r \n" %
                                            (episodes,float(agents.beta_door_value)/float(agents.beta_door_cnt),float(agents.beta_foodh)/float(agents.beta_foodh_cnt), float(agents.beta_food)/float(agents.beta_food_cnt),float(agents.beta_win)/float(agents.beta_win_cnt)))
            
            # Print out rewasympathy at door opening
            if agents.sympathetic_mode:
                log_file = open(log_dir+'reward_symp'+'.log','a')
                reward_symp_array = [float(agents.reward_foodh)/float(agents.beta_foodh_cnt), float(agents.reward_fooda)/float(agents.beta_fooda_cnt), float(agents.reward_food)/float(agents.beta_food_cnt),float(agents.reward_win)/float(agents.beta_win_cnt)]
                log_file.write("# %4d | reward: %r \n" %
                                            (episodes,list(reward_symp_array)))

            # Print out rewasympathy at door opening
            if agents.sympathetic_mode:
                log_file = open(log_dir+'reward_symp_pnext'+'.log','a')
                reward_symp_array = [float(agents.reward_foodh_pnext)/float(agents.beta_foodh_cnt),float(agents.reward_fooda_pnext)/float(agents.beta_fooda_cnt), float(agents.reward_food_pnext)/float(agents.beta_food_cnt),float(agents.reward_win_pnext)/float(agents.beta_win_cnt)]
                log_file.write("# %4d | reward: %r \n" %
                                            (episodes,list(reward_symp_array)))

            agents.cnt = 0
            agents.greedy_cost_total = 0
            agents.cost_total = 0
            agents.cnt_human = 0
            agents.human_cost_total = 0
            agents.human_perc_correct = 0
            agents.human_cost_before = 0

            agents.beta_door_value = 0
            agents.beta_door_cnt = 0
            agents.beta_food = 0
            agents.beta_food_cnt = 0
            agents.beta_foodh = 0
            agents.beta_foodh_cnt = 0
            agents.beta_fooda = 0
            agents.beta_fooda_cnt = 0
            agents.beta_win = 0
            agents.beta_win_cnt = 0
            agents.beta_step = 0
            agents.beta_step_cnt = 0
            agents.sa_reward_error = 0

            agents.reward_foodh = 0
            agents.reward_fooda = 0
            agents.reward_food = 0
            agents.reward_win = 0
            agents.reward_killed = 0
            agents.reward_harmH = 0
            agents.reward_step = 0
            agents.reward_door = 0

            agents.reward_foodh_pnext = 0
            agents.reward_food_pnext = 0
            agents.reward_win_pnext = 0
            agents.reward_killed_pnext = 0
            agents.reward_harmH_pnext = 0
            agents.reward_door_pnext = 0
            agents.reward_step_pnext = 0

    if training_mode == False:

        # delete saved file
        if (loop%agents.params['keep_interval'] != 0):
            file_location = save_dir + 'GridWorld_LearningAgent_' + str(loop)
            os.remove(file_location)
