import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import time
import argparse


def main():
    start_time_one_env = time.time()

    env = gym.make('LunarLander-v2')
    env = Monitor(env)

    model = PPO.load(args.model_file, env=env, seed=args.seed)

    time_one_env = time.time() - start_time_one_env
    print(f"learning took {time_one_env:.2f}s")

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=args.n_eval_episodes, deterministic=True, render=args.render)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Launch pybullet simulation run.')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1998)
    parser.add_argument('--n_eval_episodes', type=int, default=20)
    parser.add_argument('--model_file', type=str,
                        default="PPO_LunarLanderv2_model")

    args = parser.parse_args()
    main()
