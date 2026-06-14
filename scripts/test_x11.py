#!/usr/bin/env python3
"""X11 转发测试脚本 — 生成简单图形验证远程显示"""

import sys
import os

def test_x11():
    print("=" * 50)
    print("X11 转发测试")
    print("=" * 50)

    # 1. 检查环境变量
    display = os.environ.get("DISPLAY", "")
    print(f"\n1. DISPLAY = '{display}'")
    if not display:
        print("   ❌ DISPLAY 未设置！X11 转发未工作。")
        print("   请确保：")
        print("   - Mac 端已启动 XQuartz")
        print("   - SSH 连接时使用了 -X 参数：ssh -X zhangyx@服务器IP")
        print("   - 服务器 sshd_config 中 X11Forwarding 已启用")
        return False
    else:
        print(f"   ✅ DISPLAY 已设置为 {display}")

    # 2. 检查 XAUTHORITY
    xauthority = os.environ.get("XAUTHORITY", os.path.expanduser("~/.Xauthority"))
    print(f"\n2. XAUTHORITY = '{xauthority}'")
    if os.path.exists(xauthority):
        print(f"   ✅ Xauthority 文件存在")
    else:
        print(f"   ⚠️  Xauthority 文件不存在 (可能不影响使用)")

    # 3. 测试 tkinter
    print("\n3. 测试 tkinter...")
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # 不显示窗口
        root.destroy()
        print("   ✅ tkinter 工作正常")
    except Exception as e:
        print(f"   ❌ tkinter 错误: {e}")
        return False

    # 4. 测试 matplotlib
    print("\n4. 测试 matplotlib...")
    try:
        import matplotlib
        print(f"   backend = {matplotlib.get_backend()}")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'r-o', linewidth=2)
        ax.set_title("X11 Forwarding Test — 如果你看到这个图，说明 X11 转发成功！")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        print("   ✅ matplotlib 加载成功，尝试显示图形...")
        print("   📊 图形窗口应该出现在你的 Mac XQuartz 中！")
        plt.show()
        print("   ✅ 图形窗口已关闭")
        return True
    except Exception as e:
        print(f"   ❌ matplotlib 错误: {e}")
        return False


if __name__ == "__main__":
    success = test_x11()
    print("\n" + "=" * 50)
    if success:
        print("✅ X11 转发配置成功！")
    else:
        print("❌ X11 转发存在问题，请根据上述提示排查。")
    print("=" * 50)
    sys.exit(0 if success else 1)
