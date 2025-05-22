# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import time # Added for timing logs

import numpy as np

from lerobot.common.robot_devices.motors.configs import MujocoArmConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc # Added for timestamp logs

try:
    import mujoco
    import mujoco.viewer # Added for passive viewer
except ImportError:
    mujoco = None
    logging.warning(
        "MuJoCo not found. Please install it to use MujocoArm. "
        "Visit https://mujoco.readthedocs.io/en/latest/python.html for installation instructions."
    )


class MujocoArm:
    """
    Represents a robot arm simulated in MuJoCo.
    This class provides an interface compatible with other motor bus controllers
    (like FeetechMotorsBus or DynamixelMotorsBus) for use with ManipulatorRobot.
    """

    def __init__(self, config: MujocoArmConfig):
        if mujoco is None and not config.mock:
            raise ImportError(
                "MuJoCo is not installed, but is required for MujocoArm unless in mock mode. "
                "Please install mujoco_python_viewer or mujoco."
            )
        self.config = config
        self.model = None
        self.data = None
        self.viewer = None # Initialize viewer attribute
        self.actuator_ids = []
        self.joint_qpos_addrs = []  # For reading joint positions
        self.joint_qvel_addrs = []  # For reading joint velocities
        self.is_connected = False
        self.logs = {}  # For compatibility with other bus types

        if self.config.mock:
            num_actuators = 0
            if self.config.actuator_names:
                num_actuators = len(self.config.actuator_names)
            elif self.config.joint_names: # Should not happen with current MujocoArmConfig validation
                num_actuators = len(self.config.joint_names)
            self._mock_positions_deg = np.zeros(num_actuators)
            self._mock_velocities_deg_s = np.zeros(num_actuators)
            logging.info(f"MujocoArm '{config.xml_file_path}': Initializing in mock mode with {num_actuators} actuators.")

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"MujocoArm for {self.config.xml_file_path} is already connected."
            )

        if self.config.mock:
            self.is_connected = True
            logging.info(f"MujocoArm '{self.config.xml_file_path}': Connected in mock mode.")
            return

        if mujoco is None: # Should have been caught in __init__ if not mock
             raise RuntimeError("MujocoArm cannot connect: MuJoCo library not available.")

        try:
            self.model = mujoco.MjModel.from_xml_path(self.config.xml_file_path)
            self.data = mujoco.MjData(self.model)
            
            # Set timestep for 30Hz simulation rate
            self.model.opt.timestep = 1/30
            logging.info(f"MujocoArm '{self.config.xml_file_path}': Set model.opt.timestep to {self.model.opt.timestep:.4f}s (for 30Hz target).")

            names_to_map = self.config.actuator_names or self.config.joint_names
            if not names_to_map:
                 raise ValueError("MujocoArmConfig must have either 'actuator_names' or 'joint_names'.")


            if self.config.actuator_names:
                logging.info(f"MujocoArm '{self.config.xml_file_path}': Mapping actuator names: {self.config.actuator_names}")
                for name in self.config.actuator_names:
                    act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    if act_id == -1:
                        raise ValueError(f"MujocoArm: Actuator '{name}' not found in model '{self.config.xml_file_path}'.")
                    self.actuator_ids.append(act_id)

                    # Determine the joint controlled by this actuator to read qpos/qvel
                    # actuator_trntype can be mjOBJ_JOINT, mjOBJ_TENDON, etc.
                    # actuator_trnid is a 2-tuple (id, type_specific_info)
                    target_type = self.model.actuator_trntype[act_id]
                    target_id = self.model.actuator_trnid[act_id][0]
                    
                    if target_type == mujoco.mjtTrn.mjTRN_JOINT:
                        joint_id = target_id
                        self.joint_qpos_addrs.append(self.model.jnt_qposadr[joint_id])
                        self.joint_qvel_addrs.append(self.model.jnt_dofadr[joint_id])
                    else:
                        # If actuator doesn't directly target a joint, we can't easily get qpos/qvel for it.
                        # This might require more complex logic or different config (e.g. explicit joint names for reading).
                        raise NotImplementedError(
                            f"MujocoArm: Actuator '{name}' targets type {target_type}, not a JOINT. "
                            "Reading state for such actuators is not yet supported directly. "
                            "Please ensure actuators target joints for state reading."
                        )
            # Note: Direct joint control (using joint_names for control) is not fully fleshed out here
            # as the primary example uses position actuators. If joint_names were for control,
            # we'd map them to joint_ids and use different mjData fields for control (e.g. qfrc_applied or relying on position PIDs).

            if self.config.launch_viewer:
                logging.info(f"MujocoArm '{self.config.xml_file_path}': Launching passive MuJoCo viewer.")
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            assert len(self.config.motor_signs) == len(self.actuator_ids)
            self.motor_signs = np.array(self.config.motor_signs)
            self.is_connected = True
            logging.info(
                f"MujocoArm: Connected to model '{self.config.xml_file_path}'. "
                f"Found {len(self.actuator_ids)} actuators. "
                f"Mapped qpos addresses: {self.joint_qpos_addrs}, qvel_dof addresses: {self.joint_qvel_addrs}"
            )

        except Exception as e:
            logging.error(f"MujocoArm: Error during connect for '{self.config.xml_file_path}': {e}")
            self.model = None
            self.data = None
            self.is_connected = False
            # Do not fall back to mock automatically, let user handle/reconfigure.
            raise  # Re-raise the exception

    def disconnect(self):
        if not self.is_connected:
            # Allow calling disconnect even if not connected, similar to some other device handlers.
            logging.info(f"MujocoArm '{self.config.xml_file_path}': Was not connected or already disconnected.")
            return

        if self.viewer and self.viewer.is_running():
            self.viewer.close()
            logging.info(f"MujocoArm '{self.config.xml_file_path}': Closed MuJoCo viewer.")
            self.viewer = None

        self.model = None
        self.data = None
        self.is_connected = False
        logging.info(f"MujocoArm '{self.config.xml_file_path}': Disconnected.")

    @property
    def motor_names(self) -> list[str]:
        """Returns the names of the controlled actuators (or joints if configured for joint control)."""
        if self.config.actuator_names:
            return self.config.actuator_names
        elif self.config.joint_names: # Fallback, though current config expects actuators primarily
            return self.config.joint_names
        return []

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

    def read(self, data_name: str, motor_names: list[str] | None = None) -> np.ndarray:
        """Reads data from the simulated motors.

        Args:
            data_name: The type of data to read (e.g., "Present_Position", "Present_Speed").
            motor_names: Optional list of motor names to read from. If None, reads from all
                         configured actuators in their specified order.

        Returns:
            A numpy array containing the requested data. Values are in degrees or degrees/s.
            Returns an array of NaNs for unsupported data_name or if not connected (unless mock).
        """
        # For MujocoArm, we assume motor_names corresponds to the order in self.config.actuator_names
        # or all of them if motor_names is None. For simplicity, this implementation currently ignores
        # filtering by motor_names and always returns data for all configured actuators.
        # A more robust implementation would map motor_names to specific indices if provided.

        num_motors = len(self.config.actuator_names) if self.config.actuator_names else 0
        start_time = time.perf_counter() # Start timer

        if self.config.mock:
            if data_name == "Present_Position":
                output_values = self._mock_positions_deg.copy()
            elif data_name == "Present_Speed":
                output_values = self._mock_velocities_deg_s.copy()
            else:
                logging.debug(f"MujocoArm (Mock): Read for '{data_name}' not implemented, returning zeros.")
                output_values = np.zeros(num_motors)
            
            self.logs[f"read_{data_name}_dt_s"] = time.perf_counter() - start_time
            self.logs[f"read_{data_name}_timestamp_utc"] = capture_timestamp_utc()
            return output_values

        if not self.is_connected or not self.model or not self.data:
            logging.warning(f"MujocoArm '{self.config.xml_file_path}': Cannot read, not connected.")
            # Log before returning
            self.logs[f"read_{data_name}_dt_s"] = time.perf_counter() - start_time
            self.logs[f"read_{data_name}_timestamp_utc"] = capture_timestamp_utc()
            return np.full(num_motors, np.nan)

        if data_name == "Present_Position":
            positions_rad = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs])
            output_values = np.rad2deg(positions_rad)
            output_values = output_values * self.motor_signs
        elif data_name == "Present_Speed":
            velocities_rad_s = np.array([self.data.qvel[addr] for addr in self.joint_qvel_addrs])
            output_values = np.rad2deg(velocities_rad_s)
            output_values = output_values * self.motor_signs
        else:
            logging.warning(f"MujocoArm '{self.config.xml_file_path}': Read for data_name '{data_name}' is not implemented. Returning NaNs.")
            output_values = np.full(num_motors, np.nan)

        self.logs[f"read_{data_name}_dt_s"] = time.perf_counter() - start_time
        self.logs[f"read_{data_name}_timestamp_utc"] = capture_timestamp_utc()
        return output_values.astype(np.float32)

    def write(self, data_name: str, values: float | np.ndarray, motor_names: list[str] | None = None):
        """Writes data to the simulated motors.

        Args:
            data_name: The type of data to write (e.g., "Goal_Position").
            values: The value(s) to write. Expected in degrees for "Goal_Position".
            motor_names: Optional list of motor names to write to. If None, applies to all
                         configured actuators. (Currently ignored, applies to all).
        """
        # Similar to read, motor_names is currently ignored for simplicity.
        num_motors = len(self.config.actuator_names) if self.config.actuator_names else 0
        start_time = time.perf_counter() # Start timer

        if self.config.mock:
            if data_name == "Goal_Position":
                if isinstance(values, (float, int)):
                    self._mock_positions_deg.fill(float(values))
                elif isinstance(values, np.ndarray) and values.size == num_motors:
                    self._mock_positions_deg = values.astype(float).copy()
                else:
                    logging.warning(f"MujocoArm (Mock): Goal_Position write expects a scalar or ndarray of size {num_motors}.")
            else:
                logging.debug(f"MujocoArm (Mock): Write for '{data_name}' ignored.")
            # Log mock write attempt
            self.logs[f"write_{data_name}_dt_s"] = time.perf_counter() - start_time
            self.logs[f"write_{data_name}_timestamp_utc"] = capture_timestamp_utc()
            return

        if not self.is_connected or not self.model or not self.data:
            logging.warning(f"MujocoArm '{self.config.xml_file_path}': Cannot write, not connected.")
            self.logs[f"write_{data_name}_dt_s"] = time.perf_counter() - start_time
            self.logs[f"write_{data_name}_timestamp_utc"] = capture_timestamp_utc()
            return

        if data_name == "Goal_Position":
            if isinstance(values, (float, int)):
                # If scalar, apply to all actuators
                values_deg_arr = np.full(len(self.actuator_ids), float(values))
            elif isinstance(values, np.ndarray):
                values_deg_arr = values.astype(float)
            else:
                logging.error(f"MujocoArm '{self.config.xml_file_path}': Invalid type for Goal_Position values: {type(values)}")
                return

            if values_deg_arr.size != len(self.actuator_ids):
                logging.error(
                    f"MujocoArm '{self.config.xml_file_path}': Mismatch in Goal_Position values size. Expected {len(self.actuator_ids)}, got {values_deg_arr.size}."
                )
                return
            
            values_rad_arr = np.deg2rad(values_deg_arr)
            values_rad_arr = values_rad_arr * self.motor_signs
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = values_rad_arr[i]
            
            # Step the simulation after setting controls
            try:
                self.step_simulation()
            except Exception as e:
                logging.error(f"MujocoArm '{self.config.xml_file_path}': Error during step_simulation called from write: {e}") # Clarified error source
        elif data_name in ["Torque_Enable", "Operating_Mode"]:
            logging.debug(f"MujocoArm '{self.config.xml_file_path}': Write for data_name '{data_name}' is a no-op for simulated arm.")
        else:
            logging.warning(f"MujocoArm '{self.config.xml_file_path}': Write for data_name '{data_name}' is not implemented.")

        self.logs[f"write_{data_name}_dt_s"] = time.perf_counter() - start_time
        self.logs[f"write_{data_name}_timestamp_utc"] = capture_timestamp_utc()

    def set_calibration(self, calibration: dict):
        """Sets motor calibration data. No-op for MujocoArm.
        
        Args:
            calibration: A dictionary containing calibration data.
        """
        logging.info(f"MujocoArm '{self.config.xml_file_path}': set_calibration called, but it's a no-op for MujocoArm.")
        # For MujocoArm, units are handled internally (degrees for interface, radians for MuJoCo).
        # Physical motor calibration (like zero offsets) would be part of the MuJoCo model (XML) itself.
        pass

    def step_simulation(self):
        """Advances the MuJoCo simulation by one step and syncs the viewer."""
        if self.config.mock:
            return

        if not self.is_connected or not self.model or not self.data:
            logging.warning(f"MujocoArm '{self.config.xml_file_path}': Cannot step simulation, not connected.")
            return
        
        try:
            mujoco.mj_step(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
        except Exception as e:
            logging.error(f"MujocoArm '{self.config.xml_file_path}': Error during mj_step or viewer sync: {e}")
