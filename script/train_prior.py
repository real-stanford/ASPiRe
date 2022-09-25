import argparse
import torch
from ASPiRe.modules.network import SkillPrior, skill
import numpy as np
import wandb
import pickle
import os
import time
from torch.utils.data import DataLoader, TensorDataset
from ASPiRe.rl.component.preprocessor import vector_preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', type=int, default=0, help="which gpu")
    parser.add_argument('--d1',
                        type=str,
                        help='path to dataset',
                        default=None)
    parser.add_argument('--d2',
                        type=str,
                        help='path to dataset',
                        default=None)
    parser.add_argument('-k', type=float, default=0.01, help='kl coef')
    parser.add_argument('-latent_dim', type=int, default=10, help='latent_dim')
    parser.add_argument('-H_dim', type=int, default=10, help='H_dim')
    parser.add_argument('--beta', type=float, default=1.0, help='prior kl coef')
    parser.add_argument('--nll', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--iteration', type=int, default=int(5000))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--theta', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--past_frame', type=int, default=1)
    parser.add_argument('--prior_coef', type=float, default=1)
    parser.add_argument('--rec_coef', type=float, default=1)
    parser.add_argument('--kl_analytic', action='store_true')
    parser.add_argument('--freeze_Z', default=1000, type=int)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--name', type=str)
    parser.add_argument('--save_freq', type=int, default=100)
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda:{0}'.format(args.g))

        print("using:", device)

    save_path = os.path.expanduser(os.path.join('~/ASPiRe/skill_prior', 'maze', args.name))
    os.makedirs(save_path, exist_ok=True)

    iteration = args.iteration

    H_dim = args.H_dim
    latent_dim = args.latent_dim
    past_img_num = args.past_frame

    data_source = [args.d1, args.d2]
    state_shape = [35, 35]

    def random_noisy_vector(batch_size):
        return (np.random.rand(batch_size, 2) - 0.5) * 20

    def sample_data(sample_data_source_index, SampleFile_num=1000, EachSample_num=20):
        mix_states = []
        mix_actions = []
        for data_source_index in sample_data_source_index:
            select_data_source = data_source[data_source_index]
            files = os.listdir(select_data_source)
            file_num = len(files)
            file_index = np.random.choice(file_num, size=SampleFile_num)
            if past_img_num > 1:
                state_sample = torch.zeros(EachSample_num * SampleFile_num,
                                           past_img_num * state_shape[data_source_index])
            else:
                state_sample = torch.zeros(EachSample_num * SampleFile_num, state_shape[data_source_index])

            action_sample = torch.zeros(EachSample_num * SampleFile_num, H_dim, 2)
            for i, fi in enumerate(file_index):
                file_name = files[fi]

                infile = open(os.path.join(select_data_source, file_name), 'rb')
                dataset = pickle.load(infile)
                infile.close()

                action = np.array(dataset['actions'])[:, :2]

                if data_source_index == 0:
                    #nev
                    qpos = np.array(dataset['infos/qpos'])
                    qpos_off = qpos - np.round(qpos)
                    qvel = np.array(dataset['infos/qvel'])[:, :2] / 10
                    # qpos_diff = np.ones((qpos.shape[0], 2))
                    state = np.array(dataset['observations'])
                    bz = state.shape[0]
                    qpos_diff = random_noisy_vector(bz) / 10

                    sur = state[:, 4:].copy()
                    sur[sur == 11] = 0
                    sur[sur == 10] = 1
                    sur[sur == 12] = 2
                    for o in range(sur.shape[0]):
                        if np.random.rand() < 0.1:
                            x = np.where(sur[o] == 0)[0]
                            index = np.random.choice(len(x))
                            sur[o, index] = 3
                        else:
                            continue
                    sur = sur / 10
                    state = np.concatenate(
                        (random_noisy_vector(bz), random_noisy_vector(bz), qpos_off, qpos_diff, qvel, sur), axis=-1)
                elif data_source_index == 1:
                    #self.env._target, qpos, qpos_off, qpos_diff, qvel,sur.flatten() / 10
                    qpos = np.array(dataset['infos/qpos'])
                    #normalize speed
                    qvel = np.array(dataset['infos/qvel'])[:, :2] / 10
                    qpos_diff = (qpos[:, :2] - qpos[:, 2:]) / 10
                    state = np.array(dataset['observations'])
                    bz = state.shape[0]
                    gsur = state[:, 4:].copy()
                    # random int for sur
                    sur = np.random.randint(0, 2, size=state[:, 4:].shape)
                    # find out the frame no obstacle
                    index = (14 != gsur).all(axis=1)
                    # fill 1 for them
                    qpos_diff[index] = 1
                    # fill in obstacle
                    sur[gsur == 14] = 3
                    # normalize sur by 10
                    sur = sur / 10
                    qpos_off = qpos[:, :2] - np.round(qpos[:, :2])
                    state = np.concatenate(
                        (random_noisy_vector(bz), random_noisy_vector(bz), qpos_off, qpos_diff, qvel, sur), axis=-1)

                # Get batch of sample index
                traj_len = len(dataset['actions'])
                sample_index = np.random.randint(past_img_num, traj_len - H_dim, size=EachSample_num)

                state_sample[i * EachSample_num:(i + 1) * EachSample_num] = torch.tensor(state[sample_index],
                                                                                         dtype=torch.float32)
                action_sample[i * EachSample_num:(i + 1) * EachSample_num] = torch.stack(
                    [torch.tensor(action[j:j + H_dim, :], dtype=torch.float32) for j in sample_index])

            mix_states.append(state_sample)
            mix_actions.append(action_sample)

        mix_states = torch.cat(mix_states, dim=0)
        mix_actions = torch.cat(mix_actions, dim=0)

        return DataLoader(TensorDataset(mix_states, mix_actions),
                          batch_size=args.batch_size,
                          shuffle=True,
                          drop_last=True)

    if args.log:
        wandb.init(project='skill_prior_learning')
        wandb.config.update(args)

    primitive_skills = 2
    n_skills = primitive_skills

    spiral = SkillPrior(
        H_dim=H_dim,
        action_dim=2,
        latent_dim=latent_dim,
        hidden_dim=128,
        kl_coef=args.k,
        beta=args.beta,
        nll=args.nll,
        priors_num=n_skills,
        device=device,
    )

    if args.use_batch_norm:
        for i in range(n_skills):
            spiral.add(
                skill(
                    net_arch=[128] * 7,
                    preprocessor=vector_preprocessor(input_dim=35, fc_net_arch=[], output_dim=35),
                    action_dim=latent_dim,
                ))
    else:
        raise NotImplementedError

    spiral.to(device)

    optimizer = torch.optim.Adam(spiral.parameters(), lr=args.lr)
    prior_data_source = [[0], [1]]
    start_time = time.time()
    for iter in range(iteration):
        prior_index = iter % n_skills
        sampler = sample_data(prior_data_source[prior_index])
        ep_kl = 0
        ep_prior = 0
        ep_rec = 0

        if iter > args.freeze_Z:
            for p in spiral.skill_encoder.parameters():
                p.requires_grad = False

            for p in spiral.skill_decoder.parameters():
                p.requires_grad = False

        for state_sample, action_sample in sampler:
            state_sample = state_sample.to(device)
            action_sample = action_sample.to(device)
            output = spiral(state_sample, action_sample, prior_index=prior_index)

            loss = (args.rec_coef * output.rec_loss + output.kl_loss + args.prior_coef * output.prior_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_kl += output.kl_loss.mean().item()
            ep_prior += output.prior_loss.mean().item()
            ep_rec += output.rec_loss.mean().item()

        if args.log:
            n_sample = len(sampler)
            wandb.log({
                'kl_loss': ep_kl / n_sample,
                'rec_loss': ep_rec / n_sample,
                'ips': iter / (time.time() - start_time),
                'iter': iter,
            })
            if prior_index == 0:
                wandb.log({
                    'nev_prior_loss': ep_prior / n_sample,
                })
            elif prior_index == 1:
                wandb.log({
                    'avoid_prior_loss': ep_prior / n_sample,
                })

        if iter % args.save_freq == 0:
            torch.save(spiral, os.path.join(save_path, "checkpoint{0}.pt".format(iter // args.save_freq)))


if __name__ == "__main__":
    main()