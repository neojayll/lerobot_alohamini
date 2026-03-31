#!/usr/bin/env python
"""
单臂遥操作示例 - 使用方案2：客户端用观测值填充未控制的臂

使用方法:
    # 控制左臂
    python teleoperate_single_arm.py --remote_ip 172.18.134.136 --arm left --leader_port /dev/am_arm_leader_left

    # 控制右臂
    python teleoperate_single_arm.py --remote_ip 172.18.134.136 --arm right --leader_port /dev/am_arm_leader_right

    # 不连接机器人（调试模式）
    python teleoperate_single_arm.py --no_robot --arm left

    # 不连接主臂（仅键盘控制）
    python teleoperate_single_arm.py --no_leader --arm left
"""

import argparse
import inspect
import os
import time
import threading

# ============ WSL2 X11 Bridge for pynput ============ #
def create_x11_bridge():
    """创建一个 X11 窗口，pynput 需要窗口激活时才能监听键盘"""
    import tkinter as tk
    root = tk.Tk()
    root.title("单臂遥操作 - 点击此窗口后使用键盘控制")
    root.geometry("400x120")
    
    label = tk.Label(
        root, 
        text="请点击此窗口激活，然后使用键盘控制机器人\n按 ESC 退出",
        font=("Arial", 12)
    )
    label.pack(expand=True)
    
    def on_close():
        root.quit()
        os._exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    print("✅ X11 bridge window created successfully")
    print("   请点击窗口激活后使用键盘控制")
    root.mainloop()

# 启动 X11 bridge 窗口线程
x11_thread = threading.Thread(target=create_x11_bridge, daemon=True)
x11_thread.start()
time.sleep(0.5)
# ==================================================== #

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============ Parameter Section ============ #
parser = argparse.ArgumentParser(description="单臂遥操作控制")
parser.add_argument("--no_robot", action="store_true", help="不连接机器人，仅打印动作")
parser.add_argument("--no_leader", action="store_true", help="不连接主臂，仅使用键盘控制")
parser.add_argument("--fps", type=int, default=30, help="主循环频率 (帧每秒)")
parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi 服务器 IP 地址")
parser.add_argument(
    "--arm", 
    type=str, 
    default="left", 
    choices=["left", "right"],
    help="要控制的臂 (left 或 right)"
)
parser.add_argument("--leader_port", type=str, default="/dev/am_arm_leader_left", help="主臂设备端口")
parser.add_argument("--leader_id", type=str, default="single_arm_leader", help="主臂设备 ID")
parser.add_argument(
    "--arm_profile",
    type=str,
    default="so-arm-5dof",
    choices=["so-arm-5dof", "am-arm-6dof"],
    help="机械臂配置文件",
)

args = parser.parse_args()

NO_ROBOT = args.no_robot
NO_LEADER = args.no_leader
FPS = args.fps
CONTROL_ARM = args.arm  # "left" 或 "right"
# ========================================== #

if NO_ROBOT:
    print("🧪 NO_ROBOT 模式: 机器人不会连接，仅打印动作")

if NO_LEADER:
    print("🧪 NO_LEADER 模式: 主臂不会连接，仅使用键盘控制")

print(f"🎯 控制模式: {'左臂' if CONTROL_ARM == 'left' else '右臂'}")

# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")

# 单臂主臂配置
leader_config = SOLeaderTeleopConfig(
    port=args.leader_port,
    arm_profile=args.arm_profile,
    id=args.leader_id,
)

leader = SOLeader(leader_config)
keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
keyboard = KeyboardTeleop(keyboard_config)
robot = LeKiwiClient(robot_config)

# Connection logic
if not NO_ROBOT:
    robot.connect()
else:
    print("🧪 robot.connect() 跳过，仅打印动作")

if not NO_LEADER:
    leader.connect()
else:
    print("🧪 leader.connect() 跳过，仅打印动作")

keyboard.connect()

init_rerun(session_name="single_arm_teleop")

if not robot.is_connected or not leader.is_connected or not keyboard.is_connected:
    print("⚠️ 警告: 部分设备未连接! 仍在运行以进行调试")

# 主循环
print(f"\n🚀 开始单臂遥操作 (控制{'左' if CONTROL_ARM == 'left' else '右'}臂)")
print("   按 ESC 退出\n")

try:
    while True:
        t0 = time.perf_counter()

        # 获取观测（包含双臂当前位置）
        observation = robot.get_observation() if not NO_ROBOT else {}
        
        # 获取主臂动作
        leader_action = leader.get_action() if not NO_LEADER else {}
        
        # 方案2：用观测值填充未控制的臂
        # 1. 先构建完整的 arm_actions 字典
        arm_actions = {}

        # 添加控制臂的动作（添加 arm_{left/right}_ 前缀）
        for k, v in leader_action.items():
            arm_actions[f"arm_{CONTROL_ARM}_{k}"] = v

        # 用观测值填充另一只臂（保持当前位置）
        other_arm = "right" if CONTROL_ARM == "left" else "left"
        if not NO_ROBOT:
            for k, v in observation.items():
                if k.startswith(f"arm_{other_arm}_") and k.endswith(".pos"):
                    arm_actions[k] = v

        # 2. 键盘控制底盘和升降
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)
        lift_action = robot._from_keyboard_to_lift_action(keyboard_keys)

        # 3. 合并所有动作
        action = {**arm_actions, **base_action, **lift_action}
        
        log_rerun_data(observation, action)

        if not NO_ROBOT:
            robot.send_action(action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
        loop_dt = time.perf_counter() - t0
        loop_fps = 1.0 / loop_dt if loop_dt > 0 else float("inf")

        if NO_ROBOT:
            print(f"[fps={loop_fps:.1f}] [NO_ROBOT] action → {action}")
        else:
            # 只打印控制臂的动作，避免输出过长
            control_arm_keys = {k: v for k, v in action.items() if k.startswith(f"arm_{CONTROL_ARM}_")}
            print(f"[fps={loop_fps:.1f}] 控制{'左' if CONTROL_ARM == 'left' else '右'}臂: {control_arm_keys}")

except KeyboardInterrupt:
    print("\n⏹️ 收到退出信号...")
finally:
    print("🔧 断开连接...")
    if not NO_LEADER:
        leader.disconnect()
    if not NO_ROBOT:
        robot.disconnect()
    keyboard.disconnect()
    print("✅ 已退出")
