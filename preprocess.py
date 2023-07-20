
import os 
import shutil
import numpy as np
import torch
import gymnasium as gym
from argparse import ArgumentParser

import zarr

import trifinger_rl_datasets

from tqdm import tqdm

def quat_conjugate(a):
    return np.concatenate((-a[:3], a[-1:]), axis=-1)

quats_symmetry = np.array([[0,0,0,1],[0, 0, 0.8660254, -0.5],[0, 0, 0.8660254, 0.5]], dtype=float)
quats_symmetry_conjugate = quats_symmetry.copy()
for i in range(3):
    quats_symmetry_conjugate[i] = quat_conjugate(quats_symmetry_conjugate[i])


def unscale_transform(x, lower, upper):
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)[...,np.newaxis]
    b = np.cross(q_vec, v, axis=-1) * q_w[...,np.newaxis] * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    # np.matmul(q_vec.reshape((1, 3)), v.reshape((3, 1))).squeeze(-1) * 2.0
    return a - b + c

def quat_mul(a, b):
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    # quat = np.stack([x, y, z, w], axis=-1)
    return np.array([x,y,z,w])

# keypoints transformation: one-by-one
def transform_keypoints(quat, keypoints):
    # quat: [4], keypoints: [8,3]
    new_keypoints = []
    for keypoint in keypoints:
        new_keypoint = quat_rotate_inverse(quat, keypoint)
        new_keypoints.append(new_keypoint)
    return np.stack(new_keypoints, axis=0)


def get_obs(observation):

    # 1. to change turns
    joint_position = observation['robot_observation']['position'].reshape(3, -1) # (3,3)
    joint_velocity = observation['robot_observation']['velocity'].reshape(3, -1) # (3,3)
    joint_torque = observation['robot_observation']['torque'].reshape(3, -1) # (3,3)
    joint_fingertip_force = observation['robot_observation']['fingertip_force'].reshape(3,-1) # (3,1)
    joint_fingertip_position = observation['robot_observation']['fingertip_position'].reshape(3,-1) # (3,3)
    joint_fingertip_velocity = observation['robot_observation']['fingertip_velocity'].reshape(3,-1) # (3,3)
    last_action = observation['action'].reshape(3,-1)

    # 2. to change axis 
    object_position = observation['camera_observation']['object_position'] # [3,]
    object_orientation = observation['camera_observation']['object_orientation'] # [4,]
    object_keypoints = observation['camera_observation']['object_keypoints'] # [8,3]
    desired_keypoints = observation['desired_goal']['object_keypoints'] # [8,3]
    achieved_keypoints = observation['achieved_goal']['object_keypoints'] # [8,3]

    # 3. to change nothing
    camera_delay = observation['camera_observation']['delay'] # (1,)
    camera_confidence= observation['camera_observation']['confidence'] # (1,)
    robot_id = observation['robot_observation']['robot_id'] # (1,)

    obs_all = []
    for i in range(3):
        obs_dict = dict()

        # 1. change order
        obs_dict['joint_position'] = joint_position[[i%3, (i+1)%3, (i+2)%3]]
        obs_dict['joint_velocity'] = joint_velocity[[i%3, (i+1)%3, (i+2)%3]]
        obs_dict['joint_torque'] = joint_torque[[i%3, (i+1)%3, (i+2)%3]]
        obs_dict['joint_fingertip_force'] = joint_fingertip_force[[i%3, (i+1)%3, (i+2)%3]]
        obs_dict['joint_fingertip_position'] = np.array([quat_rotate_inverse(quats_symmetry[i], joint_fingertip_position[(i+j)%3]) for j in range(3)])
        obs_dict['joint_fingertip_velocity'] = np.array([quat_rotate_inverse(quats_symmetry[i], joint_fingertip_velocity[(i+j)%3]) for j in range(3)])
        obs_dict['last_action'] = last_action[[i%3, (i+1)%3, (i+2)%3]]

        # 2. change axis
        obs_dict['object_position'] = quat_rotate_inverse(quats_symmetry[i], object_position)
        obs_dict['object_orientation'] = quat_mul(quats_symmetry_conjugate[i], object_orientation)
        obs_dict['object_keypoints'] = transform_keypoints(quats_symmetry[i], object_keypoints)
        obs_dict['desired_keypoints'] = transform_keypoints(quats_symmetry[i], desired_keypoints)
        obs_dict['achieved_keypoints'] = transform_keypoints(quats_symmetry[i], achieved_keypoints)

        # 3. change nothing
        obs_dict['camera_delay'] = camera_delay
        obs_dict['camera_confidence'] = camera_confidence 
        obs_dict['robot_id'] = robot_id

        # flatten
        obs_vec = [] 
        for k, v in obs_dict.items():
            obs_vec.append(v.flatten())
        obs_all.append(np.concatenate(obs_vec))

    obs_all = np.stack(obs_all, axis=0)
    return obs_all


# stats = env.get_dataset_stats()
# print(stats)
# n_timesteps, obs_size, action_size

def main():

    # 1. create args
    parser = ArgumentParser()
    parser.add_argument(
        "--input-dataset",
        default="trifinger-cube-lift-real-expert-v0",
        type=str
    )
    parser.add_argument(
        "--output-dataset",
        default="trifinger-cube-lift-real-expert-v0-masa",
        type=str
    )
    args = parser.parse_args()

    # 2. create env and load dataset
    os.makedirs(f'output_datasets/{args.output_dataset}', exist_ok=True)
    env = gym.make(
            args.input_dataset,
            data_dir=f'output_datasets/{args.output_dataset}',
            flatten_obs=False)
    # M: loading all dataset is slow
    print(f"\nLoading Dataset...")
    dataset = env.get_dataset()
    # dataset = env.get_dataset(rng=(0,2))

    # 3. preprocess
    new_observations = []
    print("\nPreprocessing Dataset...")
    for obs in tqdm(dataset['observations']):
        new_obs = get_obs(obs)
        new_observations.append(new_obs)
    new_observations = np.array(new_observations)

    # 4. save
    print("\nSaving Dataset...")
    dst_dir = f'output_datasets/{args.output_dataset}/{args.input_dataset}.zarr'
    os.makedirs(f'output_datasets/{args.output_dataset}/', exist_ok=True)
    dst_store = zarr.LMDBStore(dst_dir, writemap=False)
    root = zarr.open(store=dst_store)
    root['observations'] = new_observations
    dst_store.close()

    # test
    new_env = gym.make(
            args.input_dataset, 
            flatten_obs=True,
            data_dir=f'output_datasets/{args.output_dataset}')
    new_dataset = new_env.get_dataset(rng=(0,2), clip=False)
    print('Observation Shape', new_dataset['observations'][0].shape)


if __name__ == "__main__":

    main()



