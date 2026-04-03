import numpy as np
import torch


class ReplayBuffer(object):

    def __init__(self, keys, num_envs=1024, max_size=2000):  # max_size 现在是每个环境的步数上限
        self.num_envs = num_envs  # 与您提供的数据结构匹配
        self.max_size_per_env = max_size  # 每个环境的最大步数
        self.total_buffer_capacity = self.num_envs * self.max_size_per_env  # 缓冲区总容量

        self.content = {}
        # 每一个 key 对应一个 cycle_nd_queue，其 max_size 是每个环境的容量
        for key in keys:
            self.content[key] = cycle_nd_queue(self.num_envs, self.max_size_per_env)
        # termination (done) 也是一个重要的 key
        self.content['termination'] = cycle_nd_queue(self.num_envs, self.max_size_per_env)

        self.end_idx_global = 0  # 全局结束索引，表示总共添加了多少步数据 (num_envs * steps)

    def load(self, state: np.ndarray, action: np.ndarray, termination: np.ndarray) -> None:
        """
        负责输入 state, action 和 termination 数据到 ReplayBuffer.
        Args:
            state (np.ndarray): 形状为 [512, 2000, 68]
            action (np.ndarray): 形状为 [512, 2000, 29]
            termination (np.ndarray): 形状为 [512, 2000, 1] (或 [512, 2000])
        """
        # 检查输入数据形状是否匹配预期
        if state.shape[0] != self.num_envs or state.shape[1] != self.max_size_per_env:
            raise ValueError(
                f"State shape mismatch. Expected ({self.num_envs}, {self.max_size_per_env}, 32), got {state.shape}")
        if action.shape[0] != self.num_envs or action.shape[1] != self.max_size_per_env:
            raise ValueError(
                f"Action shape mismatch. Expected ({self.num_envs}, {self.max_size_per_env}, 32), got {action.shape}")
        if termination.shape[0] != self.num_envs or termination.shape[1] != self.max_size_per_env:
            # 允许 termination 是 (512, 2000) 或 (512, 2000, 1)
            if termination.ndim == 3 and termination.shape[2] == 1:
                termination = termination.squeeze(-1)  # 移除最后一个维度
            else:
                raise ValueError(
                    f"Termination shape mismatch. Expected ({self.num_envs}, {self.max_size_per_env}, 1) or ({self.num_envs}, {self.max_size_per_env}), got {termination.shape}")

        num_steps_per_env = state.shape[1]  # 2000
        total_steps_to_add = self.num_envs * num_steps_per_env

        e_idx_global = self.end_idx_global + total_steps_to_add

        # 将数据添加到 cycle_nd_queue
        # 注意：cycle_nd_queue 的 append 方法需要原始形状的数据，它会在内部展平
        self.content['state'].append(state, self.end_idx_global, e_idx_global)
        self.content['action'].append(action, self.end_idx_global, e_idx_global)
        self.content['termination'].append(termination, self.end_idx_global, e_idx_global)  # termination 已经被 squeeze 过

        self.end_idx_global = e_idx_global

    def generate_data_loader(self, name_list: list, rollout_length: int, mini_batch_size: int, mini_batch_num: int):
        """
        生成 PyTorch DataLoader.
        Args:
            name_list (list): 要从 buffer 中提取的数据的键列表 (e.g., ['state', 'action']).
            rollout_length (int): 每个采样轨迹的长度.
            mini_batch_size (int): 每个批次中轨迹的数量.
            mini_batch_num (int): 要生成的批次总数.
        Returns:
            torch.utils.data.DataLoader: 包含采样数据的 DataLoader.
        """
        # 获取 termination 数据的展平内容，用于计算可行索引
        # 确保这里拿到的是最新的完整内容
        done_content = self.content['termination'].content
        if done_content is None:
            raise ValueError("Replay buffer is empty. Please load data first.")

        # 计算所有可行索引 (全局索引)
        feasible_indices = index_counter.calculate_feasible_index(done_content, rollout_length, self.max_size_per_env)

        # 检查是否有足够的可行索引来满足采样需求
        total_rollouts_to_sample = mini_batch_size * mini_batch_num
        if len(feasible_indices) == 0:
            print("Warning: No feasible indices available for sampling. Returning empty DataLoader.")
            # 创建一个空的 TensorDataset，以避免 DataLoader 报错
            # 确保空 Tensor 的形状与预期输出形状一致，除了批次维度和 rollout 维度
            # 假设 feature_dim 可以从 content[name].content.shape[1:] 得到
            empty_tensors = []
            for name in name_list:
                feature_shape = self.content[name].content.shape[1:]
                empty_tensors.append(torch.empty(0, rollout_length, *feature_shape))
            dataset = torch.utils.data.TensorDataset(*empty_tensors)
            data_loader = torch.utils.data.DataLoader(dataset, mini_batch_size, shuffle=False)
            return data_loader

        # 采样 rollout 的起始索引
        sampled_indices = index_counter.sample_rollout(
            feasible_indices,
            total_rollouts_to_sample,
            rollout_length
        )

        res_tensors = []
        for name in name_list:
            # 使用采样到的索引从 cycle_nd_queue 中获取数据
            # cycle_nd_queue 的 __getitem__ 会处理索引展平和形状恢复
            data_np = self.content[name][sampled_indices]
            res_tensors.append(torch.from_numpy(data_np).float())  # 转换为 PyTorch Tensor

        dataset = torch.utils.data.TensorDataset(*res_tensors)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            mini_batch_size,
            shuffle=False  # 保持采样顺序，因为是rollout
        )

        return data_loader


class index_counter():
    def __init__(self, done_flag) -> None:
        self.done_flag = done_flag
        self.cur_frame = 0

    @staticmethod
    def sample_rollout(feasible_index, batch_size, rollout_length):
        """generate index for rollout sampling

        Args:
            feasible_index (np.ndarray): please make sure [i,i+rollout_length) is useful
            batch_size (int): total number of rollouts to sample
            rollout_length (int): length of each rollout
        """
        # feasible_index 是一个 1D 数组，包含所有可行的起始索引
        # np.random.choice 默认从 1D 数组中选择
        begin_idx = np.random.choice(feasible_index, size=[batch_size, 1], replace=True)  # 允许重复采样
        bias = np.arange(rollout_length).reshape(1, -1)
        res_idx = begin_idx + bias
        return res_idx

    @staticmethod
    def calculate_feasible_index(done_flag, rollout_length, max_size_per_env):  # 添加 max_size_per_env 参数
        """
        计算可以开始一个长度为 rollout_length 的轨迹的起始索引。
        Args:
            done_flag (np.ndarray): 1D 数组，非零值表示终止。
            rollout_length (int): 轨迹长度。
            max_size_per_env (int): 每个环境的数据步数。
        Returns:
            np.ndarray: 可行的起始索引数组。
        """
        res_flag = np.ones_like(done_flag).astype(int)

        # 1. 排除包含终止的轨迹
        terminate_idx = np.where(done_flag != 0)[0].reshape(-1, 1)
        bias = np.arange(rollout_length).reshape(1, -1)
        invalid_indices_due_to_done = (terminate_idx - bias).flatten()
        invalid_indices_due_to_done = invalid_indices_due_to_done[
            (invalid_indices_due_to_done >= 0) & (invalid_indices_due_to_done < len(done_flag))]
        res_flag[invalid_indices_due_to_done] = 0

        # 2. 排除超出总缓冲区容量的轨迹
        # 任何从 len(done_flag) - rollout_length + 1 开始的索引都无法形成完整的 rollout
        res_flag[len(done_flag) - rollout_length + 1:] = 0

        # 3. 排除跨环境边界的轨迹
        num_envs = len(done_flag) // max_size_per_env

        for i in range(num_envs):
            env_start_idx = i * max_size_per_env
            # 计算在该环境内，无法形成完整 rollout (会导致跨环境) 的起始索引范围
            # 例如，如果 max_size_per_env = 2000, rollout_length = 50
            # 那么一个环境内，从索引 1951 到 1999 开始的轨迹都会跨越到下一个环境 (因为 1951 + 50 - 1 = 1999)
            # 或者说，任何 (idx % max_size_per_env) > (max_size_per_env - rollout_length) 的 idx 都是无效起始点

            # 从 env_start_idx + (max_size_per_env - rollout_length + 1) 到 env_start_idx + max_size_per_env - 1 都是无效起始点
            invalid_start_offset_in_env = max_size_per_env - rollout_length + 1

            if invalid_start_offset_in_env <= 0:
                # 如果 rollout_length 大于等于 max_size_per_env，则整个环境都不能作为起始点
                # 因为不可能在单个环境中形成 rollout
                res_flag[env_start_idx: env_start_idx + max_size_per_env] = 0
            else:
                # 否则，标记出那些离环境末尾太近而无法完成 rollout 的起始点
                res_flag[env_start_idx + invalid_start_offset_in_env: env_start_idx + max_size_per_env] = 0

        return np.where(res_flag)[0]

    @staticmethod
    def random_select(feasible_index, p=None):
        return np.random.choice(feasible_index, p=p)


class cycle_nd_queue():
    def __init__(self, num_envs=512, max_size_per_env=2000) -> None:
        self.max_size_per_env = max_size_per_env  # 这是单个环境的 max_size
        self.num_envs = num_envs
        self.total_capacity = self.num_envs * self.max_size_per_env
        self.content = None

    def append(self, item: np.ndarray, b_idx_global, e_idx_global) -> None:
        original_shape = item.shape
        if original_shape[0] != self.num_envs:
            raise ValueError(f"Item's first dimension ({original_shape[0]}) must match num_envs ({self.num_envs}).")

        # 将 item 展平为 (num_envs * num_steps, feature_dim) 或 (num_envs * num_steps,)
        if len(original_shape) > 2:  # 例如 (512, 2000, 32)
            item_flat = item.reshape(-1, original_shape[2])
        else:
            raise ValueError(f"Unsupported item shape: {original_shape}")

        current_total_steps = item_flat.shape[0]

        # 检查 item 是否太大，超过总容量
        if current_total_steps > self.total_capacity:
            print(current_total_steps, self.total_capacity)
            raise ValueError("Item is too big for the buffer total capacity.")

        if self.content is None:
            # 初始化 content 数组，形状为 (total_capacity, feature_dim) 或 (total_capacity,)
            shape = list(item_flat.shape)
            shape[0] = self.total_capacity
            self.content = np.ones(shape, dtype=item_flat.dtype) * -1  # 使用 -1 初始化，方便识别未填充部分

        # 根据全局索引计算在总内容中的起始和结束位置
        b_idx_content = b_idx_global % self.total_capacity
        e_idx_content = e_idx_global % self.total_capacity

        effective_e_idx_content = e_idx_content
        if e_idx_content == 0 and current_total_steps > 0 and b_idx_content == 0:
            effective_e_idx_content = self.total_capacity

        if b_idx_content > effective_e_idx_content:
            l = self.total_capacity - b_idx_content
            self.content[b_idx_content:] = item_flat[:l]
            self.content[:effective_e_idx_content] = item_flat[l:]
        else:
            self.content[b_idx_content:effective_e_idx_content] = item_flat

    def __getitem__(self, idx):
        # idx 可以是单个索引，也可以是 numpy 数组 (total_rollouts, rollout_length)
        if isinstance(idx, np.ndarray):
            # idx 是 (total_rollouts, rollout_length) 形状
            # 我们需要获取这些索引对应的数据
            # 将 (total_rollouts, rollout_length) 展平为 (total_rollouts * rollout_length,)
            flat_idx = idx.flatten()

            # 使用高级索引获取数据
            retrieved_data = self.content[flat_idx % self.total_capacity]

            # 恢复形状为 (total_rollouts, rollout_length, feature_dim)
            # 或 (total_rollouts, rollout_length) for termination
            if len(self.content.shape) > 1:  # 如果 content 有 feature_dim
                original_feature_dim = self.content.shape[1:]
                return retrieved_data.reshape(*idx.shape, *original_feature_dim)
            else:  # termination，没有 feature_dim
                return retrieved_data.reshape(*idx.shape)
        else:
            # 单个索引
            return self.content.__getitem__(idx % self.total_capacity)