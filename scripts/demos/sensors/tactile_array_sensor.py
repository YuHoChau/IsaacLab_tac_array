# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating the TacSL tactile sensor implementation in IsaacLab.

This script shows how to use the TactileSensor for both camera-based and force field
tactile sensing with the tactile array finger setup.

.. code-block:: bash

    # Usage
    python tactile_array_sensor.py --tactile_compliance_stiffness 30.0 --num_envs 16
    
"""

import argparse
import cv2
import math
import numpy as np
import os
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="TacSL tactile sensor example.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--tactile_compliance_stiffness",
    type=float,
    default=None,
    help="Optional: Override compliant contact stiffness (default: use USD asset values)",
)
parser.add_argument(
    "--tactile_compliant_damping",
    type=float,
    default=None,
    help="Optional: Override compliant contact damping (default: use USD asset values)",
)
parser.add_argument("--debug_sdf_closest_pts", action="store_true", help="Visualize closest SDF points.")
parser.add_argument("--debug_tactile_sensor_pts", action="store_true", help="Visualize tactile sensor points.")
parser.add_argument("--trimesh_vis_tactile_points", action="store_true", help="Visualize tactile points using trimesh.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Import our TactileSensor
from isaaclab.sensors import TactileArraySensorCfg
from isaaclab.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


@configclass
class TactileSensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with tactile sensors on the robot."""

    # Ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Robot with tactile sensor
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path="/home/yuhao/Downloads/tactile_array_finger.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            compliant_contact_stiffness=args_cli.tactile_compliance_stiffness,
            compliant_contact_damping=args_cli.tactile_compliant_damping,
            physics_material_prim_path="elastomer",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=-0.0006,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(math.sqrt(2) / 2, -math.sqrt(2) / 2, 0.0, 0.0),  # 90° rotation
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )

    # Camera configuration for tactile sensing

    # TacSL Tactile Sensor
    tactile_sensor = TactileArraySensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
        history_length=0,
        debug_vis=args_cli.debug_tactile_sensor_pts or args_cli.debug_sdf_closest_pts,
        # Sensor configuration
        enable_force_field=True,
        # Elastomer configuration
        tactile_array_size=(12, 32),
        tactile_points_distance=0.002,
        # Contact object configuration
        contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",
        # Force field physics parameters
        normal_contact_stiffness=1.0,
        friction_coefficient=2.0,
        tangential_stiffness=0.1,
        # Debug Visualization
        trimesh_vis_tactile_points=args_cli.trimesh_vis_tactile_points,
        visualize_sdf_closest_pts=args_cli.debug_sdf_closest_pts,
    )

@configclass
class NutTactileSceneCfg(TactileSensorsSceneCfg):
    """Scene with nut contact object."""

    # Nut contact object
    contact_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/contact_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_peg_8mm.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                max_angular_velocity=180.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=-0.0006,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.05, 0.023, 0.65),
            rot = (0.70710678, 0.70710678, 0.0, 0.0),
        ),
    )

def visualize_tactile_heatmap(
    tactile_data: VisuoTactileSensorData,
    nrows: int,
    ncols: int,
    env_idx: int = 0,
    use_normalized: bool = True,
    scale_factor: int = 20,
    adaptive_scale: bool = True,
):
    if tactile_data.tactile_normal_force is None:
        return

    try:
        # Get normal force data
        tactile_normal_force = tactile_data.tactile_normal_force[env_idx].cpu().numpy()
        tactile_normal_force = np.abs(tactile_normal_force)
        force_grid = tactile_normal_force.reshape(nrows, ncols)

        # Get penetration data for marking indices
        penetration = None
        if tactile_data.penetration_depth is not None:
            penetration = tactile_data.penetration_depth[env_idx].cpu().numpy()

        force_min, force_max = force_grid.min(), force_grid.max()

        # Normalize forces
        if adaptive_scale:
            force_grid = np.clip(force_grid, 0.0, None)
            max_value = force_grid.max()
            threshold = 0.0002
            
            if max_value < threshold:
                force_normalized = force_grid / 0.0008
            else:
                if max_value > 1e-9:
                    force_normalized = force_grid / max_value
                else:
                    force_normalized = np.zeros_like(force_grid)
            
            force_normalized = np.clip(force_normalized, 0.0, 1.0)
        elif use_normalized:
            force_normalized = np.clip(force_grid, 0.0, 1.0)
        else:
            if force_max > force_min:
                force_normalized = (force_grid - force_min) / (force_max - force_min)
            else:
                force_normalized = np.zeros_like(force_grid)

        # Convert to uint8 for colormap
        force_uint8 = (force_normalized * 255).astype(np.uint8)

        # Scale up by repeating pixels
        force_large = np.repeat(np.repeat(force_uint8, scale_factor, axis=0), scale_factor, axis=1)

        # Apply colormap
        heatmap = cv2.applyColorMap(force_large, cv2.COLORMAP_VIRIDIS)
        # Display heatmap only if not in headless mode
        if not args_cli.headless:
            window_name = f"Tactile Sensor Heatmap (Env {env_idx})"
            cv2.imshow(window_name, heatmap)
            cv2.waitKey(1)

    except cv2.error as e:
        if not args_cli.headless:
            print(f"[Tactile Viz] Error: {e}")
    except Exception as e:
        print(f"[Tactile Viz] Error in visualize_tactile_heatmap: {e}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Assign different masses to contact objects in different environments
    num_envs = scene.num_envs

    # Create constant downward force
    force_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    torque_tensor = torch.zeros(scene.num_envs, 1, 3, device=sim.device)
    # force_tensor[:, 0, 2] = -1.0
    # force_tensor[:, 0, 1] = -1.0

    nrows = scene["tactile_sensor"].cfg.tactile_array_size[0]
    ncols = scene["tactile_sensor"].cfg.tactile_array_size[1]

    physics_timer = Timer()
    physics_total_time = 0.0
    physics_total_count = 0

    entity_list = ["robot"]
    if "contact_object" in scene.keys():
        entity_list.append("contact_object")

    while simulation_app.is_running():

        if count == 200:
            # Reset robot and contact object positions
            count = 0
            for entity in entity_list:
                root_state = scene[entity].data.default_root_state.clone()
                root_state[:, :3] += scene.env_origins
                scene[entity].write_root_state_to_sim(root_state)

            scene.reset()
            print("[INFO]: Resetting robot and contact object state...")

        if "contact_object" in scene.keys():
            # rotation
            if count > 20:
                env_indices = torch.arange(scene.num_envs, device=sim.device)
                odd_mask = env_indices % 2 == 1
                even_mask = env_indices % 2 == 0
                torque_tensor[odd_mask, 0, 1] = 20  # rotation for odd environments
                torque_tensor[even_mask, 0, 1] = -20  # rotation for even environments
                scene["contact_object"].set_external_force_and_torque(force_tensor, torque_tensor)

        # Step simulation
        scene.write_data_to_sim()
        physics_timer.start()
        sim.step()
        physics_timer.stop()
        physics_total_time += physics_timer.total_run_time
        physics_total_count += 1
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        # Access tactile sensor data
        tactile_data = scene["tactile_sensor"].data

        # Real-time tactile visualization
        visualize_tactile_heatmap(
            tactile_data, 
            nrows, 
            ncols, 
            env_idx=0, 
            use_normalized=True, 
            adaptive_scale=True,
        )

def main():
    """Main function."""
    try:
        print("[INFO]: Initializing simulation context...")
        # Initialize simulation
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.005,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                gpu_collision_stack_size=2**30,  # Prevent collisionStackSize buffer overflow in contact-rich environments.
            ),
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        print("[INFO]: Simulation context created.")

        # Set main camera
        print("[INFO]: Setting camera view...")
        sim.set_camera_view(eye=[0.11498, 0.05623, 0.53133], target=[0.035, 0.0, 0.5])
        print("[INFO]: Camera view set.")

        # Create scene based on contact object type
        print("[INFO]: Creating scene configuration...")
        scene_cfg = NutTactileSceneCfg(num_envs=args_cli.num_envs, env_spacing=0.2)
        print("[INFO]: Initializing interactive scene...")
        scene = InteractiveScene(scene_cfg)
        print("[INFO]: Interactive scene created.")

        # Initialize simulation
        print("[INFO]: Resetting simulation...")
        sim.reset()
        print("[INFO]: Setup complete...")
        
        print("Press Ctrl+C to stop the simulation")
        print("="*70 + "\n")
        
        # Run simulation
        run_simulator(sim, scene)
    except Exception as e:
        print(f"[ERROR]: An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    finally:
        # Clean up cv2 windows (only if not in headless mode)
        if not args_cli.headless:
            cv2.destroyAllWindows()
        # Close sim app
        simulation_app.close()