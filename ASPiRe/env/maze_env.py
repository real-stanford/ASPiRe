import gym
from gym.spaces import Box, Dict
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model, maze_layouts
import numpy as np


class RandomMaze2d(gym.Env):

    def __init__(self,
                 dynamic=False,
                 maze_size=10,
                 maze_seed=0,
                 seed=None,
                 target_pos=np.array([3, 3]),
                 agent_start_pos=np.array([9, 9]),
                 chaser_start_pos=np.array([6, 6]),
                 img_size=32,
                 agent_centric=False,
                 goal_exist=True,
                 chaser_exist=False,
                 box=None,
                 box_location=None,
                 reset_box=False,
                 chaser_move=True,
                 max_episode_steps=500,
                 keep_dim=False,
                 camera_distance=5,
                 coverage_frac=0.25,
                 reward_scale=1,
                 terminate_when_hit=True,
                 hit_penalty=-10) -> None:
        super().__init__()
        self.dynamic = dynamic
        self.camera_distance = camera_distance
        self.goal_exist = goal_exist
        self.chaser_exist = chaser_exist
        self.box = box
        self.box_location = box_location
        self.reset_box = reset_box
        self.chaser_move = chaser_move
        self.agent_centric = agent_centric
        self.target_pos = target_pos
        self.agent_start_pos = agent_start_pos
        self.chaser_start_pos = chaser_start_pos
        self.maze_size = maze_size
        self.maze_seed = maze_seed
        self.seed = seed
        self.img_size = img_size
        self.keep_dim = keep_dim
        self.coverage_frac = coverage_frac
        self.hit_penalty = hit_penalty
        self.terminate_when_hit = terminate_when_hit

        if self.target_pos is None:
            self.reset_target = True
        else:
            self.reset_target = False

        if self.box_location is None:
            self.reset_box = True
        else:
            self.reset_box = False

        self.max_episode_steps = max_episode_steps
        self.ts = 0

        self.reward_scale = reward_scale

        self.env, self.controller, self.chaser_controller = self.sample_env_and_controller()

        self.observation_space = Dict()

        self.action_space = Box(low=-1, high=1, shape=(2,))

    def sample_env_and_controller(self):
        layout_str = maze_layouts.rand_layout(size=self.maze_size,
                                              seed=self.maze_seed,
                                              goal_exist=self.goal_exist,
                                              chaser_exist=self.chaser_exist,
                                              coverage_frac=self.coverage_frac,
                                              box=self.box)
        env = maze_model.MazeEnv(layout_str,
                                 reset_target=self.reset_target,
                                 reset_box=self.reset_box,
                                 reward_type='sparse',
                                 agent_centric_view=self.agent_centric,
                                 reward_scale=self.reward_scale)
        chaser_controller = None
        controller = None
        if self.chaser_exist:
            chaser_controller = waypoint_controller.WaypointController(layout_str)
        if self.goal_exist:
            controller = waypoint_controller.WaypointController(layout_str)
        return env, controller, chaser_controller

    def reset_env(self, env, agent_centric=False):
        s = env.reset()
        if self.goal_exist and self.target_pos is not None:
            env.set_target(target_location=self.target_pos)
        if self.chaser_exist:
            # chaser_start_pos = random.choice(self.chaser_start_pos_pool)
            # s = env.reset_to_location(np.concatenate([self.agent_start_pos, chaser_start_pos]))
            s = env.reset_to_location(np.concatenate([self.agent_start_pos, self.chaser_start_pos]))
        if self.agent_start_pos is not None:
            s = env.reset_to_location(self.agent_start_pos)
        if self.box_location is not None:
            env.set_box(box_location=self.box_location)
        env.set_marker()
        return s

    def reset(self):
        valid = False
        while (not valid):
            try:
                self.env, self.controller, self.chaser_controller = self.sample_env_and_controller()
                s = self.reset_env(self.env)
                position = s[0:2]
                velocity = s[2:4]
                act, done = self.controller.get_action(position, velocity, self.env._target, self.env._box)
                valid = True
            except ValueError:
                valid = False

        self.ts = 0
        if self.is_feasiable:
            return self.get_obs()
        else:
            print("goal not feasible")
            return self.reset()

    def is_feasiable(self):
        env_data = self.env.sim.data
        qpos = env_data.qpos.ravel().copy()
        qvel = env_data.qvel.ravel().copy()
        if self.goal_exist:
            try:
                act, done = self.controller.get_action(qpos[:2], qvel[:2], self.env._target)
            except ValueError:
                return False

        if self.chaser_exist:
            try:
                act, done = self.chaser_controller.get_action(qpos[2:], qvel[2:], qpos[:2])
            except ValueError:
                return False

        return True

    def gridify_state(self, state):
        return (int(round(state[0])), int(round(state[1])))

    def get_obs(self):

        env_data = self.env.sim.data
        qvel = env_data.qvel.ravel().copy()
        qvel = qvel / 10
        qpos = env_data.qpos.ravel().copy()
        int_qpos = self.gridify_state(qpos)

        sur = np.zeros((5, 5))
        for i in range(-2, 3):
            for j in range(-2, 3):
                x_index = int_qpos[0] + i
                y_index = int_qpos[1] + j
                if x_index < 0 or x_index > self.maze_size + 2 - 1 or y_index < 0 or y_index > self.maze_size + 2 - 1:
                    sur[i + 2, j + 2] = 10
                    # sur[i + 2, j + 2] = 0
                else:
                    sur[i + 2, j + 2] = self.env.maze_arr[x_index, y_index]
        box_location = self.env._box
        index = np.argmin(np.abs(box_location - qpos).sum(axis=-1))
        closest_box = box_location[index]
        qpos_diff = (qpos - closest_box) / 10
        qpos_diff = np.array([1, 1]) if (14 != sur).all() else qpos_diff

        qpos = env_data.qpos.ravel().copy()
        qpos_off = qpos - np.round(qpos)

        nev_sur_copy = sur.flatten().copy()
        # clean the obstacle to avoid OoD input to prior to free space [11]
        nev_sur_copy[nev_sur_copy == 14] = 0
        nev_sur_copy[nev_sur_copy == 11] = 0
        nev_sur_copy[nev_sur_copy == 10] = 1
        nev_sur_copy[nev_sur_copy == 12] = 2
        # nev_prior_vector = np.concatenate([qpos, qvel, nev_sur_copy], axis=-1)
        nev_prior_vector = np.concatenate([qpos_off, qvel, nev_sur_copy], axis=-1)

        avoid_sur_copy = sur.flatten().copy()
        # avoid_sur_copy[avoid_sur_copy != 0.14] = 0.11
        avoid_sur_copy[avoid_sur_copy != 14] = 0
        avoid_sur_copy[avoid_sur_copy == 14] = 1
        avoid_prior_vector = np.concatenate([qpos_diff, qvel, avoid_sur_copy], axis=-1)

        sur[sur == 11] = 0
        sur[sur == 10] = 1
        sur[sur == 12] = 2
        sur[sur == 14] = 3

        obs = {
            # 'image': None,
            'vector': np.concatenate([self.env._target, qpos, qpos_off, qpos_diff, qvel,
                                      sur.flatten() / 10], axis=-1),
            'nev_prior_vector': nev_prior_vector,
            'avoid_prior_vector': avoid_prior_vector
        }
        return obs

    def step(self, action):
        done = False
        reward = 0
        action = np.clip(action, -1, 1)

        _, reward, done, info = self.env.step(action)
        self.ts += 1

        hit_reward = 0
        push_reward = 0
        reach_reward = 0
        success = False

        if info['reach']:
            reach_reward = self.reward_scale
            done = True
            success = True

        if info['hit']:
            hit_reward = self.hit_penalty
            done = True
            success = False

        obs = self.get_obs()

        reward = reach_reward + hit_reward

        return obs, reward, done, {
            'push_reward': push_reward,
            'reach_reward': reach_reward,
            'hit_reward': hit_reward,
            'success': success,
        }
