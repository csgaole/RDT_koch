import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embodiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "data/datasets/koch/"  # 替换为实际路径
        self.DATASET_NAME = "koch_test"  # new dataset

        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths,
                                           p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(
                file_path) if not state_only else self.parse_hdf5_file_state_only(
                    file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # 根据dataset_info.json，修改此处代码
            state_seq = f['observation.state'][:]
            num_steps = state_seq.shape[0]

            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            state_delta = np.abs(state_seq - state_seq[0:1])
            indices = np.where(np.any(state_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no state that exceeds the threshold.")

            # We randomly sample a timestep
            step_id = np.random.randint(first_idx - 1, num_steps)

            # Load the instruction
            # 由于没有 language_instruction，这里设置为空字符串
            instruction = ""

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }

            # Rescale gripper to [0, 1]
            # 根据实际情况修改缩放比例
            state_seq = state_seq / np.array([1.0] * 6)
            target_qpos = f['action'][
                step_id:step_id + self.CHUNK_SIZE] / np.array([1.0] * 6)

            # Parse the state and action
            state = state_seq[step_id:step_id + 1]
            state_std = np.std(state_seq, axis=0)
            state_mean = np.mean(state_seq, axis=0)
            state_norm = np.sqrt(np.mean(state_seq**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0],
                                           1))
                ],
                                          axis=0)

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm

                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                    for i in range(5)
                ] + [STATE_VEC_IDX_MAPPING["right_gripper_joint_0_pos"]]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)

            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0),
                               step_id + 1):
                    img = f['observation.images'][key][i]
                    imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8),
                                             cv2.IMREAD_COLOR))
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE -
                                           imgs.shape[0], 1, 1, 1)),
                        imgs
                    ],
                                          axis=0)
                return imgs

            # `cam_high` is the external camera image
            cam_high = parse_img('laptop')  # 根据 dataset_info.json 修改
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1,
                            self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) +
                [True] * valid_len)
            cam_left_wrist = parse_img('phone')  # 根据 dataset_info.json 修改
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img(
                'phone1')  # 根据 dataset_info.json 修改
            cam_right_wrist_mask = cam_high_mask.copy()

            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file

        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            # 根据dataset_info.json，修改此处代码
            state_seq = f['observation.state'][:]
            num_steps = state_seq.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None

            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            state_delta = np.abs(state_seq - state_seq[0:1])
            indices = np.where(np.any(state_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no state that exceeds the threshold.")

            # Rescale gripper to [0, 1]
            state_seq = state_seq / np.array([1.0] * 6)  # 根据实际情况修改
            target_qpos = f['action'][:] / np.array(
                [1.0] * 6)  # 根据实际情况修改

            # Parse the state and action
            state = state_seq[first_idx - 1:]
            action = target_qpos[first_idx - 1:]

            # Fill the state/action into the unified vector
            def fill_in_state(values):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm

                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                    for i in range(5)
                ] + [STATE_VEC_IDX_MAPPING["right_gripper_joint_0_pos"]]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM, ))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            action = fill_in_state(action)

            # Return the resulting sample
            return True, {"state": state, "action": action}


if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)