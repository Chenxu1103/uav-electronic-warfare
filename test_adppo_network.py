#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试AD-PPO网络能否处理28维的状态输入
"""

import os
import sys
import numpy as np
import torch

print("脚本开始执行...")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"添加项目根目录到Python路径: {project_root}")

print("导入所需模块...")
from src.algorithms.ad_ppo import ActorCritic, ADPPO
from src.models.environment import ECMEnvironment

def test_actor_critic_network():
    """测试ActorCritic网络"""
    # 创建28维的假状态
    state_dim = 28
    movement_dim = 2
    jamming_dim = 4
    hidden_dim = 64
    batch_size = 10
    
    print(f"\n===== 测试ActorCritic网络 =====")
    print(f"状态维度: {state_dim}")
    print(f"移动维度: {movement_dim}")
    print(f"干扰维度: {jamming_dim}")
    
    # 创建模型
    print("创建ActorCritic模型...")
    model = ActorCritic(state_dim, movement_dim, jamming_dim, hidden_dim)
    
    # 创建随机状态
    state = torch.randn(batch_size, state_dim)
    print(f"输入状态形状: {state.shape}")
    
    # 前向传播
    try:
        print("执行前向传播...")
        result = model.forward(state)
        print("前向传播成功!")
        print(f"特征形状: {result['features'].shape}")
        print(f"移动均值形状: {result['movement_mean'].shape}")
        print(f"价值形状: {result['value'].shape}")
        
        # 测试act方法
        print("\n测试act方法...")
        state = torch.randn(1, state_dim)
        action, log_prob, value = model.act(state)
        print("act方法成功!")
        print(f"动作形状: {action.shape}")
        print(f"动作值: {action}")
        print(f"对数概率: {log_prob}")
        print(f"价值: {value}")
        
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_adppo_agent():
    """测试ADPPO智能体"""
    # 创建环境
    print("\n创建测试环境...")
    config = {
        'num_uavs': 3,
        'num_radars': 3,
        'max_steps': 200,
        'time_step': 1.0,
        'world_size': [10000, 10000, 1000],
        'target_position': np.array([8000, 8000, 100]),
        'starting_area': [0, 2000, 0, 2000, 50, 200],
        'radar_positions': [
            np.array([7000, 3000, 0]),
            np.array([5000, 5000, 0]),
            np.array([3000, 7000, 0])
        ]
    }
    
    env = ECMEnvironment(config)
    
    print(f"\n===== 测试ADPPO智能体 =====")
    
    # 获取动作维度和智能体数量
    # 实际状态维度应该是28，而不是env.observation_space.shape[0]
    state_dim = 28  
    action_dim = env.action_space.shape[0]
    num_agents = env.num_uavs
    
    print(f"设置状态维度: {state_dim}")
    print(f"环境动作维度: {action_dim}")
    print(f"智能体数量: {num_agents}")
    
    # 创建ADPPO智能体
    print("创建ADPPO智能体...")
    agent = ADPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=64,
        rollout_steps=100,
        batch_size=32
    )
    
    # 重置环境
    print("重置环境...")
    states = env.reset()
    
    # 测试选择动作
    try:
        print(f"输入状态形状: {np.array(states).shape if isinstance(states, list) else states.shape}")
        print("选择动作...")
        actions, log_probs, values = agent.select_actions(states)
        print("选择动作成功!")
        print(f"动作形状: {np.array(actions).shape}")
        print(f"动作值: {actions}")
        print(f"对数概率: {log_probs}")
        print(f"价值: {values}")
        
        # 测试环境步骤
        print("\n执行环境步骤...")
        next_states, rewards, done, info = env.step(actions)
        print("环境步骤成功!")
        print(f"奖励: {rewards}")
        print(f"完成: {done}")
        
        return True
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试AD-PPO网络和智能体...")
    
    # 测试ActorCritic网络
    print("\n运行ActorCritic网络测试...")
    network_test = test_actor_critic_network()
    
    # 测试ADPPO智能体
    print("\n运行ADPPO智能体测试...")
    agent_test = test_adppo_agent()
    
    # 总结
    print("\n=== 测试结果摘要 ===")
    if network_test and agent_test:
        print("所有测试通过!")
    elif network_test:
        print("网络测试通过，但智能体测试失败。")
    elif agent_test:
        print("智能体测试通过，但网络测试失败。")
    else:
        print("所有测试都失败了。") 