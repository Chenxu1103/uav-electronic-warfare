#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模块导入检查脚本
用于验证项目依赖和模块导入是否正常工作
"""

import os
import sys
import importlib

def check_import(module_name):
    """检查模块是否可以正确导入"""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ 成功导入模块: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {module_name}")
        print(f"   错误信息: {e}")
        return False

def main():
    # 设置项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print(f"Python路径: {current_dir}")
    
    # 检查核心依赖
    print("\n检查核心依赖:")
    dependencies = [
        "numpy", "torch", "matplotlib", "pandas", 
        "seaborn", "gym", "argparse"
    ]
    
    all_deps_ok = True
    for dep in dependencies:
        if not check_import(dep):
            all_deps_ok = False
    
    if not all_deps_ok:
        print("\n⚠️ 核心依赖检查失败，请安装缺失的依赖: pip install -r requirements.txt")
    else:
        print("\n✅ 核心依赖检查通过!")
    
    # 检查项目模块
    print("\n检查项目模块:")
    project_modules = [
        "src.models",
        "src.algorithms",
        "src.algorithms.ad_ppo",
        "src.utils"
    ]
    
    all_modules_ok = True
    for module in project_modules:
        if not check_import(module):
            all_modules_ok = False
    
    if not all_modules_ok:
        print("\n⚠️ 项目模块检查失败，请确保项目结构正确，并且Python路径设置合适。")
    else:
        print("\n✅ 项目模块检查通过!")
    
    # 项目结构检查
    print("\n检查关键文件:")
    key_files = [
        "src/main.py",
        "src/algorithms/ad_ppo.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_files_ok = True
    for file_path in key_files:
        full_path = os.path.join(current_dir, file_path)
        if os.path.isfile(full_path):
            print(f"✅ 找到文件: {file_path}")
        else:
            print(f"❌ 未找到文件: {file_path}")
            all_files_ok = False
    
    if not all_files_ok:
        print("\n⚠️ 文件检查失败，部分关键文件缺失。")
    else:
        print("\n✅ 文件检查通过!")
    
    # 总结
    if all_deps_ok and all_modules_ok and all_files_ok:
        print("\n🎉 所有检查通过! 项目结构和依赖正常。")
        print("   您可以运行以下命令开始使用:")
        print("   - 训练:   python run.py --train --algorithms ad_ppo,maddpg")
        print("   - 评估:   python run.py --evaluate --algorithms ad_ppo,maddpg")
        print("   - 可视化: python run.py --visualize --algorithms ad_ppo,maddpg")
    else:
        print("\n⚠️ 检查未完全通过，请解决上述问题后再尝试运行项目。")

if __name__ == "__main__":
    main() 