import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import Util
import colorama
import winsound
import time
import datetime

class PolicyAlgorithm(object):
    def __init__(self, env, learning_rate, discount, decay, exporation_rate, hidden_layers):
        self.learning_rate, self.init_learning = learning_rate, learning_rate
        self.discount = discount
        self.decay = decay
        self.env = env
        self.exporation_rate, self.init_exp_rate = exporation_rate, exporation_rate
        self.hidden_layers = hidden_layers
        self.num_states = env.observation_space
        self.num_actions = env.action_space
        self.model_init()
        self.reward_list = []
        self.plot_for_train = []
        
    def model_init(self):
        self.model = {
            1: np.random.randn(self.hidden_layers, self.num_states) / np.sqrt(self.num_actions) * self.learning_rate,
            2: np.random.randn(self.num_actions, self.hidden_layers) / np.sqrt(self.hidden_layers) * self.learning_rate,
        }

    def save_model(self):
        name_gen = Util.Util.cool_name_generator()
        Util.Util.save_load(0, f'saved_models\{name_gen}', self)
        append_saved_models = open('saved_models\index.txt', 'a')
        append_saved_models.write(f'{name_gen} {np.mean(self.reward_list[len(self.reward_list) - len(self.reward_list) // 100 :])}\n')
    
    def load_model(self, file_name=False):

        append_saved_models = open('saved_models\index.txt', 'r+')

        model_data = (append_saved_models.readlines()[-2]).split(" ")
        
        if not file_name:
            file_name = model_data[0]

        self = Util.Util.save_load(1, f'saved_models\{file_name}', None)

    def softmax(self, x):
        x = x.astype(np.float64)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def discount_rewards(self, rewards):
        discounted_reward = np.zeros_like(rewards, dtype=np.float64)
        total_rewards = 0
        for t in reversed(range(0, len(rewards))):
            total_rewards = total_rewards * self.discount + rewards[t]
            discounted_reward[t] = total_rewards
        return discounted_reward
    
    def policy_forward(self, state):
        #print(state.shape)
        #print(self.env.observation_space)
        m1 = self.model[1]
        m2 = self.model[2]
        h = np.dot(self.model[1], state)
        h[h<0] = 0
        logp = np.dot(self.model[2], h)
        p = self.softmax(logp)
        return p, h
    
    def policy_backward(self, processed_states, hiddle_layer_output, grad_log_prob):
        dW2 = np.dot(hiddle_layer_output.T, grad_log_prob).T
        dh = np.dot(grad_log_prob, self.model[2])
        dh[hiddle_layer_output <= 0] = 0 # backprop relu
        dW1 = np.dot(dh.T, processed_states)
        return {1:dW1, 2:dW2}
    
    def train(self, episodes, batch_size, cmax):
        start_now = datetime.datetime.now()

        for epoch in range(episodes):
            state = self.env.reset()
            prev_state = 0
            sstate, shidden, sgrads, srewards = [], [], [], []
            done = False
            counter = 0
            grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
            rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
            reward_sum = 0

            while not done and counter != cmax:
                counter += 1
                append_state = state - prev_state

                aprob, hid = self.policy_forward(append_state)
                if random.random() <= self.exporation_rate:
                    action = np.random.choice(np.arange(self.num_actions))
                else:
                    max_prob = np.max(aprob)
                    second_max_prob = np.partition(aprob, -2)[-2]
                    probability_difference = max_prob - second_max_prob

                    threshold = 0.1

                    if probability_difference > threshold:
                        action = np.random.choice(np.arange(self.num_actions), p=aprob)
                    else:
                        action = np.argmax(aprob)
                    action = np.random.choice(np.arange(self.num_actions), p=aprob)
                
                
                action_one_hot = np.zeros(self.num_actions)
                action_one_hot[action] = 1


                sstate.append(append_state)
                shidden.append(hid)
                sgrads.append(action_one_hot - aprob)
                #sgrads.append(-aprob)
                
                state, reward, done, _ = self.env.step(action)
                srewards.append(reward)

                reward_sum += reward

            vstate = np.vstack(sstate)
            vhidden = np.vstack(shidden)
            vgrads = np.vstack(sgrads)
            vrewards = np.vstack(srewards)


            discounted_vrew = self.discount_rewards(vrewards)
            discounted_vrew -= (np.mean(discounted_vrew)).astype(np.float64)
            discounted_vrew /= ((np.std(discounted_vrew)).astype(np.float64) + 1e-8)

            vgrads *= discounted_vrew
            grad = self.policy_backward(vstate, vhidden, vgrads)
            for k in self.model: 
                grad_buffer[k] = grad_buffer[k].astype(np.float64)
                grad[k] = grad[k].astype(np.float64)
                grad_buffer[k] += grad[k]
            

            self.learning_rate = self.init_learning * np.sqrt(1 - self.decay ** epoch) / (1 - self.discount ** epoch + 1e-8)
            self.exporation_rate -= self.exporation_rate / episodes
            if epoch % batch_size == 0:
                for k,v in self.model.items():
                    g = grad_buffer[k]
                    rmsprop_cache[k] = self.decay * rmsprop_cache[k] + (1 - self.decay) * g**2
                    self.model[k] += (self.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-8))
                    grad_buffer[k] = np.zeros_like(v)

            
            Util.Util.progress_bar(50 * ((epoch + 1) / episodes), 50, 'Epoch', f"Total Time Elapsed: {datetime.datetime.now() - start_now} || Reward: {reward_sum}          ")
            
            self.plot_for_train.append(epoch)
            self.reward_list.append(reward_sum)
        
        self.save_model()
        winsound.PlaySound("sgm.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
        plt.plot(np.arange(episodes), self.reward_list)
        plt.show()
    
    def render(self):
        def better_render(mogus):
            plt.cla()
            plt.plot(self.plot_for_train, self.reward_list, label='Reward')

        ani = FuncAnimation(plt.figure(), better_render, interval=100, cache_frame_data=False)
        plt.tight_layout()
        plt.show()

    def train_render(self, episodes, batch_size, cmax):
        func1 = threading.Thread(target=self.train, args=(episodes, batch_size, cmax))
        func1.start()
        self.render()
        func1.join()

    def test(self, max_count):
        state = self.env.reset()
        done = False
        counter = 0
        reward_sum = 0

        while not done and counter != max_count:
            aprob, hid = self.policy_forward(state)
            max_prob = np.max(aprob)
            second_max_prob = np.partition(aprob, -2)[-2]
            probability_difference = max_prob - second_max_prob

            threshold = 0.1

            if probability_difference > threshold:
                action = np.random.choice(np.arange(self.num_actions), p=aprob)
            else:
                action = np.argmax(aprob)
            action = np.random.choice(np.arange(self.num_actions), p=aprob)
            state, reward, done, _ = self.env.step(action, test=True)
            reward_sum += reward
            counter += 1

        print(colorama.Fore.YELLOW + 'Reward sum:', reward_sum)
        time.sleep(1)
        return counter == max_count


class TestEnv(object):
    def __init__(self):
        self.observation_space = 3
        self.action_space = 2
        self.numbers = [random.random(), random.random(), random.random()]
        self.counter = 0
    
    def reset(self):
        self.numbers = [random.random(), random.random(), random.random()]
        self.counter = 0
        return self.get_state()
            
    def get_state(self):
        self.numbers = [random.random(), random.random(), random.random()]
        return np.array(self.numbers)

    def step(self, action, test=False):
        total = np.sum(self.numbers)
        reward = 0.1 - math.fabs(total - action[0] - action[1])
        if test:
            print('Action', action)
            print('Total', total)
            print('Reward', reward)
            time.sleep(0.01)
            

        return self.get_state(), reward, self.counter == 100, None  
  
    def render(self):
        print(self.numbers, np.sum(self.numbers))
