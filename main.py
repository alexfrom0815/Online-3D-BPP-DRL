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

def main(args):
    # input arguments about environment
    config.container_size = args.bin_size
    config.box_size_set = args.item_set
    config.pallet_size = args.bin_size[0] 
    config.box_range = args.item_size_range
    config.test = (args.mode == 'test')
    config.preview = args.preview
    config.load_name = args.load_name
    config.data_name = args.data_name
    config.pretrain = args.load_model
    config.enable_rotation = args.enable_rotation

    if not config.test:
        config.data_type = args.item_seq
    config.cuda = args.use_cuda and torch.cuda.is_available()
    config.no_cuda = not config.cuda

    if config.test:
        test_model()
    else:
        train_model()

def test_model():
    assert config.test is True
    model_url = config.load_dir + config.load_name
    unified_test(model_url, config)

def train_model():
    custom = input('please input the test name: ')
    time_now = time.strftime('%Y.%m.%d-%H-%M', time.localtime(time.time()))

    env_name = config.env_name
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
    
    if config.cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(config.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:" + str(config.device) if config.cuda else "cpu")
    envs = make_vec_envs(env_name, config.seed, config.num_processes, config.gamma, config.log_dir, device, False)

    if config.pretrain:
        model_pretrained, ob_rms = torch.load(os.path.join(load_path, config.load_name))
        actor_critic = Policy(
            envs.observation_space.shape, envs.action_space,
            base_kwargs={'recurrent': config.recurrent_policy, 'hidden_size': config.hidden_size})
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
            base_kwargs={'recurrent': config.recurrent_policy, 'hidden_size': config.hidden_size})
    print(actor_critic)
    print("Rotation:", config.enable_rotation)
    actor_critic.to(device)

    # leave a backup for parameter tuning
    copyfile('config.py', os.path.join(data_path, 'config.py'))
    copyfile('main.py', os.path.join(data_path, 'main.py'))
    copyfile('./acktr/envs.py', os.path.join(data_path, 'envs.py'))
    copyfile('./acktr/distributions.py', os.path.join(data_path, 'distributions.py'))
    copyfile('./acktr/storage.py', os.path.join(data_path, 'storage.py'))
    copyfile('./acktr/model.py', os.path.join(data_path, 'model.py'))
    copyfile('./acktr/algo/acktr_pipeline.py', os.path.join(data_path, 'acktr_pipeline.py'))

    if config.algo == 'a2c':
        agent = algo.ACKTR(actor_critic,
                       config.value_loss_coef,
                       config.entropy_coef,
                       config.invalid_coef,
                       config.lr,
                       config.eps,
                       config.alpha,
                       config.max_grad_norm
                           )
    elif config.algo == 'acktr':
        agent = algo.ACKTR(actor_critic,
            config.value_loss_coef,
            config.entropy_coef,
            config.invalid_coef,
            acktr=True)

    rollouts = RolloutStorage(config.num_steps,  # forward steps
                              config.num_processes,  # agent processes
                              envs.observation_space.shape,
                              envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              can_give_up=config.give_up,
                              enable_rotation=config.enable_rotation,
                              pallet_size=config.container_size[0])

    obs = envs.reset()
    location_masks = []
    for observation in obs:
        if not config.enable_rotation:
            box_mask = get_possible_position(observation, config.container_size)
        else:
            box_mask = get_rotation_mask(observation, config.container_size)
        location_masks.append(box_mask)
    location_masks = torch.FloatTensor(location_masks).to(device)

    rollouts.obs[0].copy_(obs)
    rollouts.location_masks[0].copy_(location_masks)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_ratio = deque(maxlen=10)

    start = time.time()
    num_updates = int(config.num_env_steps) // config.num_steps // config.num_processes

    if not os.path.exists('{}/{}/{}'.format(config.tbx_dir, env_name, custom)):
        os.makedirs('{}/{}/{}'.format(config.tbx_dir, env_name, custom))
    if config.tensorboard:
        writer = SummaryWriter(logdir='{}/{}/{}'.format(config.tbx_dir, env_name, custom))

    j = 0
    index = 0
    while True:
        j += 1
        if config.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if config.algo == "acktr" else config.lr)

        for step in range(config.num_steps):
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
                if not config.enable_rotation:
                    box_mask = get_possible_position(observation, config.container_size)
                else:
                    box_mask = get_rotation_mask(observation, config.container_size)
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

        rollouts.compute_returns(next_value, False, config.gamma, 0.95, config.use_proper_time_limits)
        # value_loss, action_loss, dist_entropy, prob_loss = agent.update(rollouts)
        value_loss, action_loss, dist_entropy, prob_loss, graph_loss = agent.update(rollouts)

        rollouts.after_update()
        if config.save_model:
            if (j % config.save_interval == 0
                or j == num_updates - 1) and config.save_dir != "":
                torch.save([
                    actor_critic.state_dict(),
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(data_path, env_name + time_now + ".pt"))

        # print useful information of training
        if j % config.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config.num_processes * config.num_steps
            end = time.time()
            index += 1
            print(
                "The algorithm is {}, the recurrent policy is {}\nThe env is {}, the version is {}".format(
                    config.algo, config.recurrent_policy, env_name, custom))
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

            if config.tensorboard:
                writer.add_scalar('The average rewards', np.mean(episode_rewards), j)
                writer.add_scalar("The mean ratio", np.mean(episode_ratio), j)
                writer.add_scalar('Distribution entropy', dist_entropy, j)
                writer.add_scalar("The value loss", value_loss, j)
                writer.add_scalar("The action loss", action_loss, j)
                writer.add_scalar('Probability loss', prob_loss, j)
                writer.add_scalar("Mask loss", graph_loss, j) # add mask loss

        if (config.eval_interval is not None and len(episode_rewards) > 1
                and j % config.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, env_name, config.seed,
                     config.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    args = get_args()
    main(args)

