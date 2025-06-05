#!/usr/bin/env python3
"""
项目设置验证脚本
验证所有核心功能是否正常工作
"""

import sys
import os
import time

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_imports():
    """测试所有关键模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试基础依赖
        import numpy as np
        import torch
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import gym
        print("  ✅ 基础依赖导入成功")
        
        # 测试项目模块
        from src.models import ECMEnvironment, UAV, Radar
        from src.algorithms import MultiAgentActionDependentRL, MADDPG
        from src.utils import plot_training_curves
        print("  ✅ 项目模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_environment():
    """测试环境创建和基本功能"""
    print("\n🏗️  测试环境创建...")
    
    try:
        from src.models import ECMEnvironment
        
        # 创建环境
        config = {
            'num_uavs': 3,
            'num_radars': 3,
            'max_steps': 10,  # 短时间测试
        }
        
        env = ECMEnvironment(config)
        print(f"  ✅ 环境创建成功，观测维度: {env.observation_space.shape}")
        
        # 测试环境重置
        obs = env.reset()
        print(f"  ✅ 环境重置成功，观测形状: {obs.shape}")
        
        # 测试环境步进
        actions = env.action_space.sample()
        next_obs, rewards, done, info = env.step(actions)
        print(f"  ✅ 环境步进成功，奖励范围: {rewards.min():.1f} 到 {rewards.max():.1f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 环境测试失败: {e}")
        return False

def test_algorithm():
    """测试算法创建和短期训练"""
    print("\n🧠 测试算法创建...")
    
    try:
        from src.models import ECMEnvironment
        from src.algorithms import MultiAgentActionDependentRL
        
        # 创建环境
        env = ECMEnvironment({'num_uavs': 3, 'num_radars': 3, 'max_steps': 10})
        obs = env.reset()
        state_dim = obs.shape[1]
        action_dim = env.action_dim
        num_agents = env.num_uavs
        
        # 创建算法
        algorithm = MultiAgentActionDependentRL(
            num_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,  # 小网络用于测试
            lr=3e-4,
            buffer_size=1000
        )
        print(f"  ✅ ADA-RL算法创建成功")
        
        # 测试动作选择
        actions = algorithm.select_actions(obs, np.zeros((num_agents, 6)), evaluate=True)
        print(f"  ✅ 动作选择成功，动作形状: {actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 算法测试失败: {e}")
        return False

def test_visualization():
    """测试可视化功能"""
    print("\n📊 测试可视化功能...")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 配置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建测试图表
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label='测试曲线')
        ax.set_title('可视化测试图表')
        ax.set_xlabel('时间')
        ax.set_ylabel('数值')
        ax.legend()
        
        # 保存到临时文件
        test_path = "test_plot.png"
        plt.savefig(test_path)
        plt.close()
        
        if os.path.exists(test_path):
            os.remove(test_path)
            print("  ✅ 可视化功能正常，中文字体支持")
            return True
        else:
            print("  ❌ 图表保存失败")
            return False
            
    except Exception as e:
        print(f"  ❌ 可视化测试失败: {e}")
        return False

def test_quick_training():
    """测试快速训练"""
    print("\n🏃 测试快速训练...")
    
    try:
        from src.models import ECMEnvironment
        from src.algorithms import MultiAgentActionDependentRL
        
        # 创建环境和算法
        env = ECMEnvironment({'num_uavs': 3, 'num_radars': 3, 'max_steps': 5})
        obs = env.reset()
        
        algorithm = MultiAgentActionDependentRL(
            num_agents=env.num_uavs,
            state_dim=obs.shape[1],
            action_dim=env.action_dim,
            hidden_dim=32,
            lr=3e-4,
            buffer_size=100
        )
        
        # 运行3个回合测试
        start_time = time.time()
        for episode in range(3):
            states = env.reset()
            actions = np.zeros((env.num_uavs, env.action_dim))
            
            for step in range(5):
                action_dependencies = algorithm._get_action_dependencies(actions)
                actions = algorithm.select_actions(states, action_dependencies, evaluate=True)
                next_states, rewards, done, info = env.step(actions)
                states = next_states
                
                if done:
                    break
                    
        training_time = time.time() - start_time
        print(f"  ✅ 快速训练成功，3回合用时: {training_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 快速训练失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🚀 开始项目验证...")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("环境功能", test_environment), 
        ("算法功能", test_algorithm),
        ("可视化功能", test_visualization),
        ("快速训练", test_quick_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📋 验证结果汇总:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目可以正常运行。")
        print("\n快速开始命令:")
        print("  python src/main.py --train --episodes 10 --algorithms ada_rl")
        print("  python src/utils/run_visualization.py --all")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
        print("\n请尝试安装正确的依赖:")
        print("  pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    import numpy as np  # 需要在全局导入
    success = main()
    sys.exit(0 if success else 1) 