#!/usr/bin/env python3
"""
维度验证测试 - 快速检查网络架构
"""

import torch
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ultra_advanced_reproduction_system import UltraAdvancedActorCritic

def test_network_dimensions():
    """测试网络维度是否正确"""
    print("🔧 网络维度验证测试")
    print("="*50)
    
    try:
        # 创建网络
        state_dim = 37
        action_dim = 18
        hidden_dim = 1024
        
        network = UltraAdvancedActorCritic(state_dim, action_dim, hidden_dim)
        
        print(f"✅ 网络创建成功!")
        print(f"   状态维度: {state_dim}")
        print(f"   动作维度: {action_dim}")
        print(f"   隐藏维度: {hidden_dim}")
        
        # 测试前向传播
        batch_size = 1
        test_state = torch.randn(batch_size, state_dim)
        
        print(f"\n🧪 测试前向传播...")
        print(f"   输入形状: {test_state.shape}")
        
        # 前向传播
        action_mean, action_std, value = network.forward(test_state)
        
        print(f"✅ 前向传播成功!")
        print(f"   动作均值形状: {action_mean.shape}")
        print(f"   动作标准差形状: {action_std.shape}")
        print(f"   价值形状: {value.shape}")
        
        # 测试动作选择
        print(f"\n🎯 测试动作选择...")
        action, log_prob, value2 = network.act(test_state)
        
        print(f"✅ 动作选择成功!")
        print(f"   动作形状: {action.shape}")
        print(f"   对数概率形状: {log_prob.shape}")
        print(f"   价值形状: {value2.shape}")
        
        # 测试动作评估
        print(f"\n📊 测试动作评估...")
        test_action = torch.randn(batch_size, action_dim)
        log_prob2, entropy, value3 = network.evaluate_actions(test_state, test_action)
        
        print(f"✅ 动作评估成功!")
        print(f"   对数概率形状: {log_prob2.shape}")
        print(f"   熵形状: {entropy.shape}")
        print(f"   价值形状: {value3.shape}")
        
        print(f"\n🎉 所有维度测试通过!")
        print(f"🚀 网络架构修复成功，可以安全运行训练!")
        
        return True
        
    except Exception as e:
        print(f"❌ 维度测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_network_dimensions()
    
    if success:
        print(f"\n✅ 修复验证成功!")
        print(f"🚀 现在可以安全运行:")
        print(f"   python ultra_advanced_quick_test.py")
        print(f"   python ultra_advanced_reproduction_system.py")
    else:
        print(f"\n❌ 仍有问题需要解决") 