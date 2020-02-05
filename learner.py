import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import CartPolePolicy
from replay import Replay
from utils import DataSchema, LinearEpsilonScheduler


class CartPoleLearner:
    def __init__(
            self,
            batch_size=64,
            device='cpu',
            gamma=0.95,
            gradient_clip=0.0,
            loss_fn='L2',
    ):
        self.env = gym.make('CartPole-v0')
        self.input_size = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.device = device
        self.qnet = CartPolePolicy(self.input_size, self.num_actions, device)
        self.target_qnet = CartPolePolicy(self.input_size, self.num_actions, device)
        self.target_qnet.copy_params_(self.qnet)

        self.eps_sch = LinearEpsilonScheduler()

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=1e-4)

        if gradient_clip > 0.0:
            for p in self.qnet.parameters():
                p.register_hook(lambda grad: torch.clamp(
                    grad,
                    min=-gradient_clip,
                    max=gradient_clip
                ))

        self.schema = DataSchema(
            names=["prev_state", "action", "reward", "state", "done"],
            shapes=[(self.input_size,), (1,), (1,), (self.input_size,), (1,)],
            dtypes=[np.int64, np.int64, np.float32, np.float32, np.float32],
        )

        self.replay = Replay(100000, self.schema)

        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = loss_fn

    def update_(
            self,
            data: tuple,
            device: str = 'cpu'
    ):
        n = data[0].shape[0]
        data = [torch.from_numpy(d).to(device) for d in data]
        prev, action, reward, state, done = data

        q = self.qnet(prev)
        tq = self.target_qnet(state)

        selected_q = torch.gather(q, dim=-1, index=action)
        _, selected_tq = torch.max(tq, dim=-1, keepdim=True)

        target = reward + done * self.gamma * selected_tq

        if self.loss_fn == 'L2':
            loss = F.mse_loss(selected_q, target)
        elif self.loss_fn == 'L1':
            loss = F.l1_loss(selected_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(
            self,
            max_frame=100000,
            target_update_frequency=100,
    ):
        writer = SummaryWriter()
        episode, frame = 1, 1

        pbar = tqdm(total=max_frame)
        while frame < max_frame:
            prev_state = self.env.reset()
            episode_reward = 0

            for step in range(1, 300):
                s = torch.from_numpy(prev_state).view(-1, self.input_size).float().to(self.device)

                if random.random() < self.eps_sch.get_epsilon(frame):
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.qnet.get_greedy(s).item()

                state, reward, done, _ = self.env.step(action)

                if frame % 4 == 0:
                    data = self.replay.sample(self.batch_size)
                    self.update_(data, self.device)

                if frame % target_update_frequency == 0:
                    self.target_qnet.copy_params_(self.qnet)

                episode_reward += reward
                frame += 1
                pbar.update(1)

                if done:
                    break

                self.replay.push(prev_state, action, reward, state, done)
                prev_state = state

            writer.add_scalar('EpisodeLength', step, frame)
            writer.add_scalar('Reward', episode_reward, frame)
            writer.add_scalar('Epsilon', self.eps_sch.get_epsilon(frame), frame)

            if frame > max_frame:
                break

            episode += 1
        pbar.close()

    def save(self, data_path, episode):
        raise NotImplementedError

    def load(self, data_path, episode):
        raise NotImplementedError

    def play(self, num_episodes=10):
        raise NotImplementedError
