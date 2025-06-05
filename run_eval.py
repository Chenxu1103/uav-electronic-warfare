import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.environment.electronic_warfare_env import ElectronicWarfareEnv
from src.algorithms.ad_ppo import ADPPO
from src.algorithms.maddpg import MADDPG
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="ad_ppo", choices=["ad_ppo", "maddpg"], help="Algorithm to use")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--num_uavs", type=int, default=3, help="Number of UAVs")
    parser.add_argument("--num_radars", type=int, default=3, help="Number of radars")
    parser.add_argument("--env_size", type=float, default=2000.0, help="Environment size")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    parser.add_argument("--visualize", action="store_true", help="Visualize the environment")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--save_results", action="store_true", help="Save evaluation results")
    parser.add_argument("--save_path", type=str, default="eval_results", help="Path to save results")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--override_state_dim", type=int, default=None, help="手动指定状态维度，覆盖环境返回的维度")
    return parser.parse_args()

def evaluate(env, agent, args):
    total_rewards = []
    jamming_ratios = []
    success_rates = []
    min_distances = []
    total_steps = 0
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        min_dist = float('inf')
        
        print(f"\n===== Episode {episode+1} =====")
        
        while not done and episode_steps < args.max_steps:
            # 确保动作是NumPy数组
            action = agent.select_action(obs, deterministic=True)
            
            # 如果动作是元组（如果网络返回了额外信息），只使用第一个元素
            if isinstance(action, tuple):
                action = action[0]
            
            # 确保是numpy数组
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            obs = next_obs
            episode_steps += 1
            total_steps += 1
            
            if 'min_distance' in info:
                min_dist = min(min_dist, info['min_distance'])
            
            if args.verbose and episode_steps % 50 == 0:
                print(f"Step {episode_steps}, Reward: {reward:.2f}")
            
            if args.visualize and episode_steps % 5 == 0:
                env.render()
                time.sleep(0.1)  # 延迟以便观察
        
        # 获取评估指标
        jamming_ratio = info.get('jamming_ratio', 0)
        success = info.get('success', False)
        
        total_rewards.append(episode_reward)
        jamming_ratios.append(jamming_ratio)
        success_rates.append(1 if success else 0)
        min_distances.append(min_dist)
        
        print(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Steps: {episode_steps}")
        print(f"雷达干扰率: {jamming_ratio*100:.2f}%")
        print(f"本回合最小距离: {min_dist}")
        print("===== 雷达状态详情 =====")
        env.print_radar_status()
        print("\n===== UAV状态详情 =====")
        env.print_uav_status()
        
        if args.visualize:
            plt.pause(1)
    
    # 计算平均指标
    avg_reward = np.mean(total_rewards)
    avg_jamming_ratio = np.mean(jamming_ratios)
    success_rate = np.mean(success_rates) * 100
    avg_min_distance = np.mean(min_distances)
    
    print("\n===== 评估结果汇总 =====")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均雷达干扰率: {avg_jamming_ratio*100:.2f}%")
    print(f"平均最小距离: {avg_min_distance:.2f}")
    
    if args.save_results:
        os.makedirs(args.save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(args.save_path, f"{args.algorithm}_eval_{timestamp}.txt")
        
        with open(result_file, 'w') as f:
            f.write(f"Algorithm: {args.algorithm}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Episodes: {args.num_episodes}\n")
            f.write(f"Average Reward: {avg_reward:.2f}\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Average Jamming Ratio: {avg_jamming_ratio*100:.2f}%\n")
            f.write(f"Average Minimum Distance: {avg_min_distance:.2f}\n")
            f.write("\nEpisode Details:\n")
            for i in range(args.num_episodes):
                f.write(f"Episode {i+1}: Reward={total_rewards[i]:.2f}, Jamming={jamming_ratios[i]*100:.2f}%, Success={success_rates[i]}, MinDist={min_distances[i]:.2f}\n")
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'avg_jamming_ratio': avg_jamming_ratio,
        'avg_min_distance': avg_min_distance
    }

def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 添加打印方法到环境
    def print_radar_status():
        for i, radar in enumerate(env.radars):
            status = "已干扰" if radar.is_jammed else "未干扰"
            print(f"雷达 {i+1}: 位置 {radar.position}, 干扰状态: {status}, 干扰功率: {radar.jamming_power:.4f}")
    
    def print_uav_status():
        for i, uav in enumerate(env.uavs):
            # 计算与每个雷达的距离
            distances = [np.linalg.norm(uav.position - radar.position) for radar in env.radars]
            closest_radar_idx = np.argmin(distances)
            
            status = "开启" if uav.is_jamming else "关闭"
            print(f"UAV {i+1}: 位置 {uav.position}, 能量: {uav.energy:.2f}, 干扰状态: {status}, 与雷达{closest_radar_idx+1}最近, 距离: {distances[closest_radar_idx]:.2f}")
    
    # 根据模型路径判断是哪个阶段的模型
    stage = None
    if "stage" in args.model_path:
        try:
            # 从路径提取阶段信息，如stage1, stage2等
            stage_str = os.path.basename(os.path.dirname(args.model_path))
            if stage_str.startswith("stage"):
                stage = int(stage_str[5:])
        except:
            pass
    
    # 根据不同阶段配置环境
    num_radars = args.num_radars
    if stage is not None:
        print(f"检测到阶段{stage}的模型")
        if stage == 1:
            num_radars = 1  # 阶段1: 1个雷达
            print("使用阶段1配置: 1个雷达")
        elif stage == 2:
            num_radars = 1  # 阶段2: 1个雷达，大环境
            print("使用阶段2配置: 1个雷达，大环境")
        elif stage == 3:
            num_radars = 2  # 阶段3: 2个雷达
            print("使用阶段3配置: 2个雷达")
        elif stage == 4:
            num_radars = 3  # 阶段4: 3个雷达
            print("使用阶段4配置: 3个雷达")
    
    # 创建环境
    env = ElectronicWarfareEnv(
        num_uavs=args.num_uavs,
        num_radars=num_radars,  # 使用根据阶段确定的雷达数量
        env_size=args.env_size,
        max_steps=args.max_steps,
        dt=args.dt,
    )
    
    # 添加打印方法到环境
    env.print_radar_status = print_radar_status
    env.print_uav_status = print_uav_status
    
    # 创建智能体
    if args.algorithm == "ad_ppo":
        # 使用环境的状态维度或手动指定的维度
        if args.override_state_dim is not None:
            state_dim = args.override_state_dim
            print(f"使用手动指定的状态维度: {state_dim}")
        else:
            state_dim = env.observation_space.shape[0]
            
        action_dim = env.action_space.shape[0]
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        agent = ADPPO(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
        )
    elif args.algorithm == "maddpg":
        if hasattr(env, 'get_agent_obs_dim'):
            obs_dims = [env.get_agent_obs_dim() for _ in range(args.num_uavs)]
            action_dims = [env.get_agent_action_dim() for _ in range(args.num_uavs)]
        else:
            # 如果环境没有定义获取维度的方法，使用默认值
            obs_dim = env.observation_space.shape[0] // args.num_uavs
            action_dim = env.action_space.shape[0] // args.num_uavs
            obs_dims = [obs_dim] * args.num_uavs
            action_dims = [action_dim] * args.num_uavs
        
        agent = MADDPG(
            obs_dims=obs_dims,
            action_dims=action_dims,
            hidden_dim=args.hidden_dim,
            n_agents=args.num_uavs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    try:
        agent.load(args.model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 评估模型
    print(f"\n开始评估 {args.algorithm} 算法...")
    evaluate(env, agent, args)
    
    # 关闭环境
    if hasattr(env, 'close'):
        env.close()

if __name__ == "__main__":
    main() 