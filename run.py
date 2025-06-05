#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目启动脚本
用于设置正确的Python路径并启动项目
"""

import os
import sys
import argparse

def main():
    # 设置项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print(f"设置项目路径: {current_dir}")
    
    # 创建实验结果目录
    os.makedirs(os.path.join(current_dir, 'experiments', 'results'), exist_ok=True)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多无人机电子对抗决策算法")
    parser.add_argument("--train", action="store_true", help="训练算法")
    parser.add_argument("--evaluate", action="store_true", help="评估算法")
    parser.add_argument("--visualize", action="store_true", help="可视化结果")
    parser.add_argument("--episodes", type=int, default=300, help="训练回合数")
    parser.add_argument("--max_steps", type=int, default=200, help="每回合最大步数")
    parser.add_argument("--save_path", type=str, default="experiments/results", help="保存路径")
    parser.add_argument("--algorithms", type=str, default="ad_ppo,maddpg", 
                        help="要使用的算法，用逗号分隔，可选: ada_rl, maddpg, ad_ppo")
    parser.add_argument("--ppo_iterations", type=int, default=100, help="AD-PPO训练迭代次数")
    
    args = parser.parse_args()
    
    # 构建命令行参数字符串
    cmd_args = ""
    if args.train:
        cmd_args += " --train"
    if args.evaluate:
        cmd_args += " --evaluate"
    if args.visualize:
        cmd_args += " --visualize"
    
    cmd_args += f" --episodes {args.episodes}"
    cmd_args += f" --max_steps {args.max_steps}"
    cmd_args += f" --save_path {os.path.join(current_dir, args.save_path)}"
    cmd_args += f" --algorithms {args.algorithms}"
    cmd_args += f" --ppo_iterations {args.ppo_iterations}"
    
    # 导入并执行main.py
    print(f"启动程序，参数: {cmd_args}")
    print("=" * 50)
    
    try:
        import src.main
        sys.argv = sys.argv[:1] + cmd_args.split()
        src.main.main()
    except ImportError as e:
        print(f"错误: 无法导入主程序模块 (src.main)")
        print(f"错误详情: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 