from typing import Optional, Callable, Tuple, Dict, Any

import jax
import jax.numpy as jp
import mujoco.mjx as mjx

from brax.envs.base import Env, State, Wrapper
from mujoco_playground._src import mjx_env, wrapper


class VmapWrapper(Wrapper):
    r"""
    Vectorizes Brax env.
    `brax/envs/wrappers/training.py - VmapWrapper`
    """

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array, trajectory_data) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset, in_axes=(0, None))(rng, trajectory_data)

    def step(self, state: State, action: jax.Array, trajectory_data) -> State:
        return jax.vmap(self.env.step, in_axes=(0, 0, None))(
            state, action, trajectory_data
        )

    def _get_motor_targets(self, state: State, action: jax.Array, use_residual_action: bool, trajectory_data) -> jax.Array:
        return jax.vmap(self.env._get_motor_targets, in_axes=(0, 0, None, None))(
            state, action, use_residual_action, trajectory_data
        )


class ModifiedEpisodeWrapper(Wrapper):
    r"""
    Maintains episode step count and sets done at episode end.
    `brax/envs/wrappers/training.py - EpisodeWrapper`
    """

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array, trajectory_data) -> State:
        # rng: (num_envs, 2)
        # -> state [dict]: (num_envs, ...)
        state = self.env.reset(rng, trajectory_data)

        # (num_envs,)
        state.info["steps"] = jp.zeros(rng.shape[:-1])
        state.info["truncation"] = jp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info["episode_done"] = jp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics["sum_reward"] = jp.zeros(rng.shape[:-1])
        episode_metrics["average_sum_reward"] = jp.zeros(rng.shape[:-1])
        episode_metrics["length"] = jp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
            episode_metrics["average_" + metric_name] = jp.zeros(rng.shape[:-1])
        state.info["episode_metrics"] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array, trajectory_data) -> State:
        def f(state, _):
            nstate = self.env.step(state, action, trajectory_data)
            return nstate, nstate.reward

        # state [mjx_env.State]: (num_envs, ...)
        # -> state [mjx_env.State]: (num_envs, ...)
        # -> rewards [jax.Array]: (action_repeat, num_envs)
        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))        # (num_envs,)
        steps = state.info["steps"] + self.action_repeat
        done = state.done
        state.info["steps"] = steps

        # NOTE: 
        # In current `env.step()` implementation, if any envs are done in NOW step, they are reset in `env.step()` (w/o wrapper). Their `done, reward, metrics` are NOT reset/overwritten (so `done=1`), `episode_metrics` are NOT modified,
        #    and only `data, info, obs` are overwritten.
        # So, for the envs that are done and reset, `state.info` here is the `info` after the last step and after resetting; for the envs that are not done, `state.info` is the `info` after last step. And `prev_done` here includes 
        #    only the envs that were done in last step.
        # Therefore:
        #    1. `*= (1-prev_done)` is needed to zero out the `episode_metrics` of the envs that were done in last step, so that they won't accumulate metrics any more.
        #       Note that this leads to 1-action-step-lag to the real training situation: Envs rollout for 1 action step before their metrics are cleared.
        #       Note that this only affects metrics logging (metrics at (k+1)-step reflects step k's situation), and does harm to training.
        #    2. If `action_repeat > 1`, the `length` of envs that were done in the middle will still increase by `action_repeat` (instead of some value < `action_repeat`).

        # Aggregate state metrics into episode metrics
        prev_done = state.info["episode_done"]                                      # done of previous `step()` calling
        state.info["episode_metrics"]["sum_reward"] += jp.sum(rewards, axis=0)      # sum of envs
        state.info["episode_metrics"]["length"] += self.action_repeat
        state.info["episode_metrics"]["average_sum_reward"] = (
            state.info["episode_metrics"]["sum_reward"]
            / state.info["episode_metrics"]["length"]
        )

        # In our implementation, all reward terms are included in state.metrics
        for metric_name in state.metrics.keys():
            if metric_name != "reward":
                state.info["episode_metrics"][metric_name] += state.metrics[metric_name]    # sum across episode steps (action steps)
                state.info["episode_metrics"]["average_" + metric_name] = (
                    state.info["episode_metrics"][metric_name]
                    / state.info["episode_metrics"]["length"]
                )                                                                   # mean across episode steps (action steps)
                state.info["episode_metrics"][metric_name] *= 1 - prev_done
        
        state.info["episode_metrics"]["sum_reward"] *= 1 - prev_done
        state.info["episode_metrics"]["length"] *= 1 - prev_done
        state.info["episode_done"] = done
        return state


class ModifiedDomainRandomizationVmapWrapper(Wrapper):
    r"""
    Brax wrapper for domain randomization.
    `mujoco_playground/_src/wrapper.py - DomainRandomizationVmapWrapper`

    NOTICE: Original `DomainRandomizationVmapWrapper` inherits from `mujoco_playground._src.wrapper.Wrapper`, but not from `brax.envs.base.Wrapper`.
            However, this only affects whether the return value of `step()` and `reset()` is of type `brax.envs.base.State` or `mujoco_playground.mjx_env.MjxEnv`.
            Thus, the implementation below does not affect correctness.
    """

    def __init__(
        self,
        env: mjx_env.MjxEnv,
        randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
    ):
        super().__init__(env)
        self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

    def _env_fn(self, mjx_model: mjx.Model) -> mjx_env.MjxEnv:
        env = self.env
        env.unwrapped._mjx_model = mjx_model
        return env

    def reset(self, rng: jax.Array, trajectory_data) -> mjx_env.State:
        def reset(mjx_model, rng, trajectory_data):
            env = self._env_fn(mjx_model=mjx_model)
            return env.reset(rng, trajectory_data)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0, None])(
            self._mjx_model_v, rng, trajectory_data
        )
        return state

    def step(
        self, state: mjx_env.State, action: jax.Array, trajectory_data
    ) -> mjx_env.State:
        def step(mjx_model, s, a, trajectory_data):
            env = self._env_fn(mjx_model=mjx_model)
            return env.step(s, a, trajectory_data)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0, None])(
            self._mjx_model_v, state, action, trajectory_data
        )
        return res

    def _get_motor_targets(self, state: State, action: jax.Array, use_residual_action: bool, trajectory_data) -> jax.Array:
        return jax.vmap(self.env._get_motor_targets, in_axes=(0, 0, None, None))(
            state, action, use_residual_action, trajectory_data
        )


class SampleClusterMotionWrapper(Wrapper):
    r"""
    Pair traj_id of each env with cluster id assigned by dagger_cfg.
    """
    def __init__(
        self,
        env: mjx_env.MjxEnv,
    ):
        super().__init__(env)
        self.traj_sample_cluster_ids = env.traj_sample_cluster_ids
        assert self.traj_sample_cluster_ids is not None
        self.traj_sample_cluster_ids = jp.array(self.traj_sample_cluster_ids, dtype=jp.int32)

    def reset(self, rng: jax.Array, trajectory_data) -> State:
        r"""
        NOTE: we use `state.info["traj_no"]` (curr step traj_no) instead of `state.info["traj_info"].traj_state.traj_no` (next step traj_no)
        since start & end pose of every trajectory are similar, while a traj might be ended in the middle.
        we use curr step traj_no to pair with the teacher to avoid OOD
        """
        state = self.env.reset(rng, trajectory_data)
        state.info["ref_motion_cluster_id"] = jax.vmap(
            lambda traj_id: self.traj_sample_cluster_ids[traj_id]
        )(state.info["traj_no"])

        return state

    def step(self, state: mjx_env.State, action: jax.Array, trajectory_data) -> State:
        r"""
        NOTE: we use `state.info["traj_no"]` (curr step traj_no) instead of `state.info["traj_info"].traj_state.traj_no` (next step traj_no)
        since start & end pose of every trajectory are similar, while a traj might be ended in the middle.
        we use curr step traj_no to pair with the teacher to avoid OOD
        """
        res = self.env.step(state, action, trajectory_data)
        res.info["ref_motion_cluster_id"] = jax.vmap(
            lambda traj_id: self.traj_sample_cluster_ids[traj_id]
        )(res.info["traj_no"])

        return res


def wrap_fn(
        
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[[mjx.Model], tuple[mjx.Model, mjx.Model]] | None = None,
) -> wrapper.Wrapper:
    """Common wrapper pattern for all brax training agents.

    Args:
      env: environment to be wrapped
      vision: whether the environment will be vision based
      num_vision_envs: number of environments the renderer should generate,
        should equal the number of batched envs
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized model
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
      
    NOTICE: Original `wrap_fn` is in `mujoco_playground/_src/wrapper.py - wrap_for_brax_training()`, and it has an additional wrapping step `BraxAutoResetWrapper`
    """

    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = ModifiedDomainRandomizationVmapWrapper(env, randomization_fn)
    env = ModifiedEpisodeWrapper(env, episode_length, action_repeat)
    env = SampleClusterMotionWrapper(env)
    
    return env
