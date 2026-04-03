import inspect
import functools
import time
import os
import pytz

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from absl import logging
from typing import Any, Callable, Optional, Tuple
import tqdm
import tyro
import wandb
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

from brax.training.agents.ppo.networks import make_ppo_networks

import track_mj as tmj
from track_mj import update_file_handler
from track_mj.constant import WANDB_PATH_LOG
from track_mj.envs.g1_tracking.train.base_env import G1Env
from track_mj.learning.policy.ppo import train_tracking as ppo
from track_mj.envs.g1_tracking.utils.wrapper import wrap_fn
from track_mj.dr.domain_randomize_tracking import (
    domain_randomize,
    domain_randomize_terrain,
)

@dataclass
class Args:
    task: str
    exp_name: str = "debug"
    exp_tags: str = None
    exp_notes: str = None
    seed: int = 42
    convert_onnx: bool = True
    
    # ====== policy ======
    num_timesteps: int = 2_000_000_000

    obs_noise_level: float = 1.0
    history_len: int = 0


def _prepare_exp_name(task: str, exp_name: str) -> str:
    r"""
    timestamp_task_expname
    """
    cst_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    timestamp = cst_time.strftime("%m%d%H%M")
    return f"{timestamp}_{task}_{exp_name}"

def _parse_exp_tags(tags):
    r"""
    Parse tags like `"'[tag1, tag2]'" into a list.
    """
    if isinstance(tags, list):
        return tags
    if isinstance(tags, str):
        cleaned = tags.strip()
        if (cleaned.startswith('[') and cleaned.endswith(']')) or \
           (cleaned.startswith('(') and cleaned.endswith(')')) or \
           (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        # Handle quoted tags
        result = []
        for tag in cleaned.split(','):
            tag = tag.strip()
            if tag.startswith('"') and tag.endswith('"') or \
               tag.startswith("'") and tag.endswith("'"):
                tag = tag[1:-1]
            if tag:  # Ensure no empty tags are added
                result.append(tag)
        return result
    return [str(tags)]

def _validate_exp_name_format(exp_name: str, debug_mode: bool):
    if not debug_mode and len(exp_name.split("_")) != 4:
        raise ValueError(f"exp_name should be in the format <task>_<tag>_<version>, got {exp_name}")


def _apply_policy_args_to_config(args: Args, cfg, debug: bool):
    cfg.num_timesteps = args.num_timesteps
    if debug:
        cfg.training_metrics_steps = 1000
        cfg.num_evals = 0           # NOTE: not implemented. 2: init eval & last eval
        cfg.batch_size = 8
        cfg.num_minibatches = 2
        cfg.num_envs = cfg.batch_size * cfg.num_minibatches
        cfg.episode_length = 200
        cfg.unroll_length = 10
        cfg.num_updates_per_batch = 1
        cfg.action_repeat = 1
        cfg.num_timesteps = 100_000
        cfg.num_resets_per_eval = 1

def _apply_env_args_to_config(args: Args, cfg):
    cfg.history_len = args.history_len
    cfg.noise_config.level = args.obs_noise_level

    cfg.obs_keys = sorted(list(set(cfg.obs_keys)))
    cfg.privileged_obs_keys = sorted(list(set(cfg.privileged_obs_keys)))

    print("Final obs keys:", cfg.obs_keys)
    print("Final privileged obs keys:", cfg.privileged_obs_keys)


def _enable_debug_mode():
    jax.config.update("jax_traceback_filtering", "off")
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)

def _setup_paths(exp_name: str) -> tuple[Path, Path]:
    logdir = Path(WANDB_PATH_LOG) / "track" / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    update_file_handler(filename=f"{logdir}/info.log")
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    return logdir, ckpt_path

def _log_checkpoint_path(ckpt_path: Path):
    logging.info(f"Checkpoint path: {ckpt_path}")

def _prepare_training_params(cfg, ckpt_path: Path):
    params = cfg.to_dict()
    params.pop("network_factory", None)
    params["wrap_env_fn"] = wrap_fn
    network_fn = make_ppo_networks
    params["network_factory"] = (
        functools.partial(network_fn, **cfg.network_factory) if hasattr(cfg, "network_factory") else network_fn
    )
    params["save_checkpoint_path"] = ckpt_path
    return params

def _init_wandb(args: Args, exp_name, env_class, task_cfg, ckpt_path, config_fname="config.json"):
    wandb.init(
        name=exp_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group="Track",
        config={
            "num_timesteps": args.num_timesteps,
            "task": args.task,
            "group": "Track",
        },
        dir=os.path.join(WANDB_PATH_LOG, "track"),
        tags=_parse_exp_tags(args.exp_tags),
        notes=args.exp_notes,
    )
    wandb.config.update(task_cfg.to_dict())
    wandb.save(inspect.getfile(env_class))
    config_path = ckpt_path / config_fname
    config_path.write_text(task_cfg.to_json_best_effort(indent=4))


def _progress(num_steps, metrics, times, total_steps, debug_mode):
    r"""
    Log metrcis to wandb. Estimate remaining time.

    Args:
        num_steps (int):current number of steps
        metrics (dict): metrics to log
        times (list): list of time stamps
        total_steps: int, total number of steps
        debug_mode: bool, whether in debug mode
    """
    now = time.monotonic()
    times.append(now)
    if metrics and not debug_mode:
        try:
            wandb.log(metrics, step=num_steps)
        except Exception as e:
            logging.warning(f"wandb.log failed: {e}")

    if len(times) < 2 or num_steps == 0:
        return
    step_times = np.diff(times)
    median_step_time = np.median(step_times)
    if median_step_time <= 0:
        return
    steps_logged = num_steps / len(step_times)
    est_seconds_left = (total_steps - num_steps) / steps_logged * median_step_time
    logging.info(f"NumSteps {num_steps} - EstTimeLeft {est_seconds_left:.1f}[s]")

def _report_training_time(times):
    if len(times) > 1:
        logging.info("Done training.")
        logging.info(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        logging.info(f"Time to train: {times[-1] - times[1]:.2f}s")


def get_trajectory_handler(env, args: Args):
    # load reference trajectory
    trajectory_data = env.prepare_trajectory(env._config.reference_traj_config.name)
    obs_size = env.observation_size
    act_size = env.action_size
    env.th.traj = None

    # output the dataset and observation info of general tracker
    print("=" * 50)
    print(
        f"Tracking {len(trajectory_data.split_points) - 1} trajectories with {trajectory_data.qpos.shape[0]} timesteps, fps={1 / env.dt:.1f}"
    )
    print(f"Observation: {env._config.obs_keys}")
    print(f"Privileged state: {env._config.privileged_obs_keys}")
    print("=" * 50)

    return trajectory_data, obs_size, act_size


def train(args: Args):
    env_class = tmj.registry.get(args.task, "tracking_train_env_class")
    task_cfg = tmj.registry.get(args.task, "tracking_config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    exp_name = _prepare_exp_name(args.task, args.exp_name)
    debug_mode = "debug" in exp_name

    if debug_mode:
        _enable_debug_mode()

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    _apply_policy_args_to_config(args, policy_cfg, debug_mode)
    _apply_env_args_to_config(args, env_cfg)

    if args.task == "G1TrackingGeneralTerrainDR":
        hfield_data = jp.asarray(np.load("storage/data/hfield/terrain.npz")["hfield_data"])
        policy_cfg.randomization_fn = functools.partial(domain_randomize_terrain, all_hfield_data=hfield_data)
        del hfield_data
        assert env_cfg.terrain_type == "rough_terrain"
    elif args.task == "G1TrackingGeneralDR":
        assert policy_cfg.randomization_fn == domain_randomize
    elif args.task == "G1TrackingGeneral":
        assert policy_cfg.randomization_fn == None

    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    if not debug_mode:
        _init_wandb(args, exp_name, env_class, task_cfg, ckpt_path)

    train_fn = functools.partial(ppo.train, **policy_params)
    times = [time.monotonic()]      # global 

    env: G1Env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    # _eval_env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)

    trajectory_data, obs_size, act_size = get_trajectory_handler(env, args)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        trajectory_data=trajectory_data,
        progress_fn=lambda s, m: _progress(
            num_steps=s,
            metrics=m,
            times=times,
            total_steps=policy_cfg.num_timesteps,
            debug_mode=debug_mode
        ),
        policy_params_fn=lambda *args: None,
    )

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    # eval_env = env_class(terrain_type=env_cfg.terrain_type, config=env_cfg)
    # _run_evaluation(args.task, task_cfg, eval_env, inference_fn, debug_mode)
    logging.info(f"Run {exp_name} Train done.")

    if args.convert_onnx:
        env.prepare_trajectory(env._config.reference_traj_config.name)

        try:
            from track_mj.eval.tracking.brax2onnx import convert_jax2onnx, get_latest_ckpt

            ckpt_dir = get_latest_ckpt(ckpt_path)
            policy_obs_key = policy_cfg.network_factory.policy_obs_key
            convert_jax2onnx(
                ckpt_dir=ckpt_dir,
                output_path=f"{ckpt_dir}/policy.onnx",
                inference_fn=inference_fn,
                hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                obs_size=obs_size,
                action_size=act_size,
                policy_obs_key=policy_obs_key,
                jax_params=params,
                activation="swish",
            )
        except ImportError:
            logging.warning("TensorFlow is not installed. Please install TensorFlow to use ONNX conversion.")


if __name__ == "__main__":
    train(tyro.cli(Args))
