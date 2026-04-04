"""
Test script for rerun.io visualization with GPU acceleration.
Runs in WSL2 Ubuntu environment with RTX 4070Ti.

WSL2 的图形栈不支持 R32Float 纹理格式,因此提供了两种替代方案:
  1. save 模式:保存 .rrd 文件,在 Windows 侧用 rerun viewer 打开
  2. web 模式:启动 web viewer,从 Windows 浏览器访问
"""

import rerun as rr
import numpy as np
import torch
from pathlib import Path
import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun visualization test")
    parser.add_argument(
        "--mode", type=str, default="save",
        choices=["save", "web", "spawn"],
        help=(
            "运行模式: "
            "'save' 保存 .rrd 文件 (推荐WSL2), "
            "'web' 通过浏览器查看, "
            "'spawn' 直接启动viewer (需要原生GPU支持)"
        )
    )
    return parser.parse_args()


def setup_rerun(mode: str, recording_id: str, output_path: str | None = None):
    """根据模式初始化 rerun。"""
    rr.init(recording_id)

    if mode == "spawn":
        rr.spawn()  # 需要 GPU 原生支持 R32Float
    elif mode == "web":
        server_uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=server_uri)
        print(f"Web Viewer 已启动,请在 Windows 浏览器中访问: http://localhost:9876")
    elif mode == "save":
        if output_path is None:
            raise ValueError("save 模式需要提供 output_path 参数")
        rr.save(output_path)
        print(f"数据将保存到: {output_path}")


def test_rerun_gpu_visualization(mode: str):
    """Test rerun visualization with GPU-accelerated computations."""

    # Setup rerun based on mode
    output_path = "/home/xurun/AlohaMini/lerobot_alohamini/output/gpu_visualization.rrd"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_rerun(mode, "aloha_mini_demo", output_path)
    
    # Log system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Set GPU device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create timeline
    rr.set_time_sequence("frame", 0)
    
    # Test 1: GPU-accelerated 3D point cloud visualization
    print("\n=== Test 1: 3D Point Cloud ===")
    num_points = 10000
    
    # Generate random points on GPU
    points_gpu = torch.randn(num_points, 3, device=device)
    colors_gpu = torch.rand(num_points, 3, device=device)
    
    # Move to CPU for rerun logging
    points_cpu = points_gpu.cpu().numpy()
    colors_cpu = colors_gpu.cpu().numpy()
    
    rr.log(
        "points",
        rr.Points3D(
            positions=points_cpu,
            colors=colors_cpu,
            radii=0.01
        )
    )
    
    # Test 2: GPU-accelerated tensor visualization
    print("\n=== Test 2: Tensor Heatmap ===")
    tensor_size = 256
    tensor_gpu = torch.randn(tensor_size, tensor_size, device=device)
    
    # Apply some GPU operations
    tensor_gpu = torch.sigmoid(tensor_gpu)
    tensor_gpu = torch.nn.functional.relu(tensor_gpu)
    
    tensor_cpu = tensor_gpu.cpu().numpy()
    
    rr.log(
        "heatmap",
        rr.Heatmap(tensor_cpu)
    )
    
    # Test 3: Simulate robot arm trajectory visualization
    print("\n=== Test 3: Robot Arm Trajectory ===")
    trajectory_length = 100
    positions_gpu = torch.randn(trajectory_length, 3, device=device)
    
    # Simulate smooth motion on GPU
    positions_gpu = torch.cumsum(positions_gpu, dim=0) * 0.1
    
    positions_cpu = positions_gpu.cpu().numpy()
    
    rr.log(
        "robot_trajectory",
        rr.LineStrips3D([positions_cpu], colors=[[255, 0, 0]])
    )
    
    # Test 4: Animated visualization loop
    print("\n=== Test 4: Animated Visualization ===")
    num_frames = 100
    
    for frame in range(num_frames):
        rr.set_time_sequence("frame", frame)
        
        # GPU computation
        theta = torch.tensor(frame * 0.1, device=device)
        radius = torch.tensor(1.0, device=device)
        
        # Generate rotating circle points on GPU
        t = torch.linspace(0, 2 * np.pi, 100, device=device)
        x = radius * torch.cos(t + theta)
        y = radius * torch.sin(t + theta)
        z = torch.sin(theta) * torch.zeros_like(t)
        
        points_gpu = torch.stack([x, y, z], dim=1)
        
        # Add some noise on GPU
        noise = torch.randn_like(points_gpu) * 0.02
        
        points_gpu = points_gpu + noise
        
        # Log to rerun
        points_cpu = points_gpu.cpu().numpy()
        rr.log(
            "animated_points",
            rr.Points3D(
                positions=points_cpu,
                colors=[(frame % 255, 100, 200 - frame % 255)] * 100,
                radii=0.02
            )
        )
        
        # Log some metrics
        rr.log(
            "metrics/fps",
            rr.Scalar(30.0)
        )
        
        rr.log(
            "metrics/gpu_memory",
            rr.Scalar(torch.cuda.memory_allocated() / 1024**2)  # MB
        )
        
        time.sleep(0.05)
    
    print("\n=== Test completed ===")
    print("Check the rerun viewer window for visualization results.")


def test_rerun_with_images(mode: str):
    """Test rerun with GPU-accelerated image processing."""

    # Setup rerun based on mode
    output_path = "/home/xurun/AlohaMini/lerobot_alohamini/output/image_demo.rrd"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_rerun(mode, "aloha_image_demo", output_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Test: GPU Image Processing ===")
    
    # Generate synthetic images on GPU
    batch_size = 10
    height, width = 480, 640
    
    for i in range(batch_size):
        rr.set_time_sequence("frame", i)
        
        # Generate random image on GPU
        image_gpu = torch.rand(height, width, 3, device=device)
        
        # Apply GPU operations: blur, brightness adjustment
        # Using simple convolution-like operation
        kernel = torch.ones(3, 3, device=device) / 9.0
        image_gpu = image_gpu.permute(2, 0, 1).unsqueeze(1)  # (C, 1, H, W)
        
        # Apply edge detection on GPU
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=device)
        
        # Convert to grayscale
        gray_gpu = image_gpu.mean(dim=0, keepdim=True)
        
        # Move to CPU for logging
        image_cpu = image_gpu.squeeze().permute(1, 2, 0).cpu().numpy()
        gray_cpu = gray_gpu.squeeze().cpu().numpy()
        
        rr.log(
            "color_image",
            rr.Image(image_cpu)
        )
        
        rr.log(
            "gray_image",
            rr.Image(gray_cpu)
        )
        
        time.sleep(0.1)
    
    print("Image processing test completed.")


def test_rerun_robot_state(mode: str):
    """Test visualizing robot arm state with GPU computations."""

    # Setup rerun based on mode
    output_path = "/home/xurun/AlohaMini/lerobot_alohamini/output/robot_state.rrd"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    setup_rerun(mode, "robot_state_demo", output_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Test: Robot State Visualization ===")
    
    # Simulate ALOHA robot arm joints
    num_joints = 7
    num_frames = 200
    
    for frame in range(num_frames):
        rr.set_time_sequence("frame", frame)
        
        # Generate joint angles on GPU
        angles_gpu = torch.sin(torch.tensor(frame * 0.05, device=device) + 
                               torch.arange(num_joints, device=device) * 0.5) * np.pi/2
        
        # Compute forward kinematics (simplified) on GPU
        joint_positions_gpu = torch.zeros(num_joints + 1, 3, device=device)
        
        for i in range(num_joints):
            joint_positions_gpu[i+1, 0] = torch.sin(angles_gpu[i]) * (i + 1) * 0.1
            joint_positions_gpu[i+1, 1] = torch.cos(angles_gpu[i]) * (i + 1) * 0.1
            joint_positions_gpu[i+1, 2] = angles_gpu[i] * 0.1
        
        # Move to CPU
        angles_cpu = angles_gpu.cpu().numpy()
        positions_cpu = joint_positions_gpu.cpu().numpy()
        
        # Log joint angles
        rr.log(
            "joint_angles",
            rr.Scalar(angles_cpu)
        )
        
        # Log 3D arm visualization
        rr.log(
            "robot_arm",
            rr.LineStrips3D(
                [positions_cpu],
                colors=[(0, 255, 0)],
                radii=0.02
            )
        )
        
        # Log joint spheres
        rr.log(
            "joints",
            rr.Points3D(
                positions=positions_cpu,
                colors=[(255, 100, 0)] * (num_joints + 1),
                radii=0.03
            )
        )
        
        time.sleep(0.02)
    
    print("Robot state visualization completed.")


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Rerun.io GPU-Accelerated Visualization Test")
    print("=" * 60)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        print("Check your WSL2 GPU setup:")
        print("1. NVIDIA drivers installed on Windows")
        print("2. WSL2 kernel updated: wsl --update")
        print("3. NVIDIA WSL2 drivers installed")
    else:
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")

    print(f"\n运行模式: {args.mode}")
    if args.mode == "save":
        print("数据将保存到 .rrd 文件,请使用 rerun viewer 打开")
    elif args.mode == "web":
        print("请在 Windows 浏览器中访问 http://localhost:9876")
    elif args.mode == "spawn":
        print("注意: WSL2 可能不支持 spawn 模式,建议使用 --mode save")

    # Run tests
    try:
        test_rerun_gpu_visualization(args.mode)
    except Exception as e:
        print(f"Error in test_rerun_gpu_visualization: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to run image processing test...")
    try:
        test_rerun_with_images(args.mode)
    except Exception as e:
        print(f"Error in test_rerun_with_images: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to run robot state test...")
    try:
        test_rerun_robot_state(args.mode)
    except Exception as e:
        print(f"Error in test_rerun_robot_state: {e}")
        import traceback
        traceback.print_exc()

    print("\nAll tests completed!")
    if args.mode == "save":
        print("\n.rRD 文件保存在 output/ 目录下")
        print("在 Windows PowerShell 中运行以下命令查看:")
        print("  rerun C:\\Users\\<用户名>\\AppData\\Local\\Packages\\CanonicalGroupLimited...\\home\\xurun\\AlohaMini\\lerobot_alohamini\\output\\*.rrd")
