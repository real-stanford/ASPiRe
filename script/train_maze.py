from ASPiRe.rl.utility.helper import set_seed
import argparse
from ASPiRe.env.maze_env import RandomMaze2d
from ASPiRe.rl.agents.composite_ac import CompositeSAC
from ASPiRe.rl.component.preprocessor import dict_preprocessor
from d4rl.locomotion.wrappers import NormalizedBoxEnv
from ASPiRe.rl.component.normalizer import Dummay_Normalizer, Normalizer, Dict_Normalizer

import wandb
from ASPiRe.rl.component.prior_policy import prior_actor
from ASPiRe.rl.component.replay_buffer import DictReplayBuffer

from ASPiRe.rl.component.weight_policy import AdaptiveWeight
from ASPiRe.env.llc_wrapper_env import llc_wrapper_env

from gym.wrappers import TimeLimit
import torch
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, default=0, help="which gpu")
    parser.add_argument('--num_box', type=int, default=5)
    parser.add_argument('--hit_penalty', type=float, default=-0.05)
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--prior_name', type=str, default=None)
    parser.add_argument('--prior_checkpoint', type=int, default=-1)
    parser.add_argument('--target_kl', type=float, default=12, help='target kl')
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--alpha_scheduler', action='store_true')
    parser.add_argument('--prior_exploration', action='store_true')
    parser.add_argument('--prior_exploration_step', type=int, default=2000)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--target_entropy', type=float, default=-10)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--difficulty', type=int, default=2, help="goal/box position")
    parser.add_argument('--exe_prior_index', type=int, default=-1, help="execute prior index")
    parser.add_argument('--update_start', type=int, default=1)
    parser.add_argument('--theta', type=float, default=1, help="theta")
    parser.add_argument('--lamda', type=float, default=1, help="lamda")
    parser.add_argument('--update_method', type=str, default='ac')
    parser.add_argument('--weight_policy', type=str, default='softmax')
    parser.add_argument('--raw_kl', action='store_true')
    parser.add_argument('--include_entropy', action='store_true')
    parser.add_argument('--analytic_kl', action='store_true')
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--weight_use_batch_norm', action='store_true')
    parser.add_argument('--freeze_batch_norm', action='store_true')
    parser.add_argument('--Algo', type=str, default='Aspire')
    parser.add_argument('--clip_kl_divergence', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min_coef', default=1e-3, type=float)
    parser.add_argument('--n_samples', default=20, type=int)
    parser.add_argument('--max_updates', default=1e4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--max_action_range', default=2, type=float)
    parser.add_argument('--weight_lr', default=3e-4, type=float)
    parser.add_argument('--policy_reuse', action='store_true')
    parser.add_argument('--eps', default=0, type=float)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    wandb.init(project="camera-ready")
    wandb.config.update(args)

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{0}'.format(args.g))
        print("using:", device)
    if args.difficulty == 1:
        start_pos = np.array([2, 10])
        target_pos = np.array([8, 1])
        box_pos = np.array([[4, 6.], [8., 6.], [6., 2.], [2, 9.], [1., 3.]])
        vector_input_dim = 35
        dynamic_maze = False,
    elif args.difficulty == 2:
        start_pos = None
        target_pos = None
        box_pos = None
        dynamic_maze = True
        vector_input_dim = 35
    elif args.difficulty == 3:
        start_pos = np.array([2, 10])
        target_pos = np.array([8, 1])
        box_pos = np.array([[4, 6.]])
        vector_input_dim = 35
        dynamic_maze = False,

    pretain_spiral_path = os.path.expanduser(
        "~/ASPiRe/skill_prior/maze/{0}/checkpoint{1}.pt".format(
            args.prior_name, args.prior_checkpoint))
    pretain_spiral = torch.load(pretain_spiral_path, map_location=device)
    pretain_spiral.device = device
    pretain_spiral = pretain_spiral.eval()
    priors = pretain_spiral.skill_priors[:2]
    for p in pretain_spiral.parameters():
        p.requires_grad = False
    # action decoder
    action_decoder = pretain_spiral.skill_decoder
    action_decoder = action_decoder.to(device)
    action_decoder.eval()


    for p in action_decoder.parameters():
        p.requires_grad = False

    env = TimeLimit(
        llc_wrapper_env(NormalizedBoxEnv(
            RandomMaze2d(dynamic=dynamic_maze,
                         maze_size=10,
                         maze_seed=0,
                         img_size=64,
                         target_pos=target_pos,
                         chaser_start_pos=None,
                         agent_start_pos=start_pos,
                         agent_centric=True,
                         goal_exist=True,
                         chaser_exist=False,
                         box=args.num_box,
                         box_location=box_pos,
                         chaser_move=False,
                         max_episode_steps=args.max_steps,
                         keep_dim=False,
                         reward_scale=args.reward_scale,
                         hit_penalty=args.hit_penalty)),
                        H_dim=10,
                        action_dim=10,
                        skills_decoder=action_decoder,
                        device=device), args.max_steps)

    env.seed(args.seed)

    preprocessor = dict_preprocessor(input_dim=vector_input_dim, entry='vector')
    weight_preprocessor = dict_preprocessor(input_dim=vector_input_dim, entry='vector')

    if args.exe_prior_index >= 0:
        execute_prior = True
    else:
        execute_prior = False
    prior_policy = prior_actor(
        net_arch=[128] * 7,
        action_dim=10,
        max_action_range=args.max_action_range,
        tanh_squash_porb=True,
        preprocessor=preprocessor,
        priors=priors,
        execute_prior=execute_prior,
        execute_prior_index=args.exe_prior_index,
        use_batch_norm=args.use_batch_norm,
        freeze_batch_norm=args.freeze_batch_norm,
    )

    weight_policy = AdaptiveWeight(
        net_arch=[128] * 7 if args.weight_use_batch_norm else [128] * 3,
        weight_dim=2,
        action_dim=10,
        preprocessor=weight_preprocessor,
        temperature=args.temperature,
        update_start=args.update_start,
        policy=args.weight_policy,
        use_batch_norm=args.weight_use_batch_norm,
        min_coef=args.min_coef,
        max_updates=args.max_updates,
        device=device,
    )

    alpha_scheduler_fn = None

    sac_agent = CompositeSAC(
        policy=prior_policy,
        gamma=args.gamma,
        weight_policy=weight_policy,
        env=env,
        learning_starts=10,
        target_kl=args.target_kl,
        target_entropy=args.target_entropy,
        replay_buffer_class=DictReplayBuffer,
        replay_buffer_args={
            "action_dim":
                10,
            "obs_entry_info": [('vector', (vector_input_dim,)), ('nev_prior_vector', (vector_input_dim - 6,)),
                               ('avoid_prior_vector', (vector_input_dim - 6,))]
        },
        replay_buffer_size=int(1e6),
        normalizer=Dict_Normalizer(entry_info=[('vector',
                                                (vector_input_dim,))]) if args.normalize else Dummay_Normalizer(),
        log=True,
        run_name=args.name,
        checkpoint_frequency=args.save_freq,
        alpha_schedule_fn=alpha_scheduler_fn,
        prior_exploration=args.prior_exploration,
        prior_exploration_steps=args.prior_exploration_step,
        theta=args.theta,
        lamda=args.lamda,
        update_method=args.update_method,
        raw_kl=args.raw_kl,
        include_entropy=args.include_entropy,
        analytic_kl=args.analytic_kl,
        weight_network=args.weight_policy,
        clip_kl_divergence=args.clip_kl_divergence,
        n_samples=args.n_samples,
        max_action_range=args.max_action_range,
        weight_lr=args.weight_lr,
        policy_reuse=args.policy_reuse,
        eps=args.eps,
        device=device)

    sac_agent.learn(total_timesteps=1e6, log_interval=100)


if __name__ == "__main__":
    main()