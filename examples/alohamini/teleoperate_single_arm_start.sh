#!/bin/bash
# 单臂遥操作启动脚本
#
# 使用方法:
#   ./teleoperate_single_arm_start.sh
#
# 可选参数（修改下方变量）:
#   REMOTE_IP     - 服务器 IP 地址，默认 127.0.0.1
#   ARM           - 控制的臂，可选 left 或 right，默认 left
#   LEADER_PORT   - 主臂设备端口，默认 /dev/am_arm_leader_left
#   ARM_PROFILE   - 机械臂配置，可选 so-arm-5dof 或 am-arm-6dof，默认 so-arm-5dof
#   NO_ROBOT      - 调试模式：不连接机器人，设为 "--no_robot" 启用
#   NO_LEADER     - 调试模式：不连接主臂，设为 "--no_leader" 启用
#
# 示例：
#   # 控制右臂
#   修改 ARM="right" 和 LEADER_PORT="/dev/am_arm_leader_right"
#
#   # 调试模式（不连接机器人）
#   修改 NO_ROBOT="--no_robot"

# Windows端需要执行以下操作先把usb设备共享给wsl2
# usbipd list 查看usb设备清单
# usbipd attach --wsl --busid 2-2 把usb设备2-2共享给wsl2

# export WGPU_BACKEND=vulkan 如果界面无法显示则使用这个参数

# ============ 默认参数 ============ #
REMOTE_IP="10.1.1.59"
ARM="left"
LEADER_PORT="/dev/ttyACM0"
ARM_PROFILE="so-arm-5dof"
NO_ROBOT=""
NO_LEADER=""
# ================================= #

echo "=========================================="
echo "  单臂遥操作控制"
echo "=========================================="
echo "服务器 IP: $REMOTE_IP"
echo "控制臂: ${ARM}"
echo "主臂端口: $LEADER_PORT"
echo "机械臂配置: $ARM_PROFILE"
echo "=========================================="

cd "$(dirname "$0")"

python teleoperate_single_arm.py \
    --remote_ip "$REMOTE_IP" \
    --arm "$ARM" \
    --leader_port "$LEADER_PORT" \
    --arm_profile "$ARM_PROFILE" \
    $NO_ROBOT \
    $NO_LEADER
