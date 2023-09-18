import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import time
from collections import deque
import numpy as np
import torch
from shutil import copyfile
import config
from acktr import algo, utils
from acktr.utils import get_possible_position, get_rotation_mask
from acktr.envs import make_vec_envs
from acktr.arguments import get_args
from acktr.model import Policy
from acktr.storage import RolloutStorage
from evaluation import evaluate
from tensorboardX import SummaryWriter
from unified_test import unified_test
from gym.envs.registration import register

def main(args):
    # input arguments about environment
    config.pallet_size = args.bin_size[0]
    config.test = (args.mode == 'test')
    config.load_name = args.load_name
    config.data_name = args.data_name

    if config.test:
        test_model(args)
    else:
        train_model(args)

def test_model(args):
    assert config.test is True
    model_url = config.load_dir + config.load_name
    unified_test(model_url, config)

def train_model(args):
    custom = input('please input the test name: ')
    time_now = time.strftime('%Y.%m.%d-%H-%M', time.localtime(time.time()))

    env_name = args.env_name
    torch.cuda.set_device(config.device)
    # set random seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    save_path = config.save_dir
    load_path = config.load_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(load_path):
        os.makedirs(load_path)
    data_path = os.path.join(save_path, custom)
    try:
        os.makedirs(data_path)
    except OSError:
        pass

    log_dir = './log'  # directory to save agent logs (default: ./log)
    log_dir = os.path.expanduser(log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device(args.device)
    envs = make_vec_envs(env_name, config.seed, args.num_processes, args.gamma, log_dir, device, False, args = args)

    if args.pretrain:
        model_pretrained, ob_rms = torch.load(os.path.join(load_path, config.load_name))
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
        load_dict = {k.replace('module.', ''): v for k, v in model_pretrained.items()}
        load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
        load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}
        for k, v in load_dict.items():
            if len(v.size()) <= 3:
                load_dict[k] = v.squeeze(dim=-1)
        actor_critic.load_state_dict(load_dict)
        setattr(utils.get_vec_normalize(envs), 'ob_rms', ob_rms)
    else:
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size,'args': args})
    print(actor_critic)
    print("Rotation:", args.enable_rotation)
    actor_critic.to(device)

    # leave a backup for parameter tuning
    copyfile('config.py', os.path.join(data_path, 'config.py'))
    copyfile('main.py', os.path.join(data_path, 'main.py'))
    copyfile('./acktr/envs.py', os.path.join(data_path, 'envs.py'))
    copyfile('./acktr/distributions.py', os.path.join(data_path, 'distributions.py'))
    copyfile('./acktr/storage.py', os.path.join(data_path, 'storage.py'))
    copyfile('./acktr/model.py', os.path.join(data_path, 'model.py'))
    copyfile('./acktr/algo/acktr_pipeline.py', os.path.join(data_path, 'acktr_pipeline.py'))

    if args.algorithm == 'a2c':
        agent = algo.ACKTR(actor_critic,
                       args.value_loss_coef,
                       args.entropy_coef,
                       args.invalid_coef,
                       args.lr,
                       args.eps,
                       args.alpha,
                       max_grad_norm = 0.5
                           )
    elif args.algorithm == 'acktr':
        agent = algo.ACKTR(actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            args.invalid_coef,
            acktr=True)

    rollouts = RolloutStorage(args.num_steps,  # forward steps
                              args.num_processes,  # agent processes
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=False,
                              enable_rotation=args.enable_rotation,
                              pallet_size=args.container_size[0])

    obs = envs.reset()
    location_masks = []
    for observation in obs:
        if not args.enable_rotation:
            box_mask = get_possible_position(observation, args.container_size)
        else:
            box_mask = get_rotation_mask(observation, args.container_size)
        location_masks.append(box_mask)
    location_masks = torch.FloatTensor(location_masks).to(device)

    rollouts.obs[0].copy_(obs)
    rollouts.location_masks[0].copy_(location_masks)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_ratio = deque(maxlen=10)

    start = time.time()

    tbx_dir = './runs'
    if not os.path.exists('{}/{}/{}'.format(tbx_dir, env_name, custom)):
        os.makedirs('{}/{}/{}'.format(tbx_dir, env_name, custom))
    if args.tensorboard:
        writer = SummaryWriter(logdir='{}/{}/{}'.format(tbx_dir, env_name, custom))

    j = 0
    index = 0
    while True:
        j += 1
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], location_masks)

            location_masks = []
            obs, reward, done, infos = envs.step(action)
            for i in range(len(infos)):
                if 'episode' in infos[i].keys():
                    episode_rewards.append(infos[i]['episode']['r'])
                    episode_ratio.append(infos[i]['ratio'])
            for observation in obs:
                if not args.enable_rotation:
                    box_mask = get_possible_position(observation, args.container_size)
                else:
                    box_mask = get_rotation_mask(observation, args.container_size)
                location_masks.append(box_mask)
            location_masks = torch.FloatTensor(location_masks).to(device)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks, location_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, False, args.gamma, 0.95, False)
        # value_loss, action_loss, dist_entropy, prob_loss = agent.update(rollouts)
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)

        rollouts.after_update()
        if args.save_model:
            if (j % args.save_interval == 0) and config.save_dir != "":
                torch.save([
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(data_path, env_name + time_now + ".pt"))

        # print useful information of training
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            index += 1
            print(
                "The algorithm is {}, the recurrent policy is {}\nThe env is {}, the version is {}".format(
                    args.algorithm, False, env_name, custom))
            print(
                "Updates {}, num timesteps {}, FPS {} \n"
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                "The dist entropy {:.5f}, The value loss {:.5f}, the action loss {:.5f}\n"
                "The mean space ratio is {}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss, np.mean(episode_ratio)))

            if args.tensorboard:
                writer.add_scalar('The average rewards', np.mean(episode_rewards), j)
                writer.add_scalar("The mean ratio", np.mean(episode_ratio), j)
                writer.add_scalar('Distribution entropy', dist_entropy, j)
                writer.add_scalar("The value loss", value_loss, j)
                writer.add_scalar("The action loss", action_loss, j)
                writer.add_scalar('Probability loss', prob_loss, j)
                writer.add_scalar("Mask loss", graph_loss, j) # add mask loss


def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )


if __name__ == "__main__":
    registration_envs()
    args = get_args()
    main(args)

