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

import threading
import time

import numpy as np

from lerobot.common.robot_devices.cameras.configs import MujocoCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)

import mujoco
from lerobot.common.utils.utils import capture_timestamp_utc


class MujocoCamera:
    def __init__(self, config: MujocoCameraConfig):
        self.config = config
        self.model = None
        self.data = None

        self.width = config.width
        self.height = config.height
        self.channels = config.channels
        self.fps = config.fps

        self.renderer: mujoco.Renderer | None = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}

    def set_mj_model(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"MujocoCamera(camera_name='{self.config.camera_name}', camera_id={self.config.camera_id}) is already connected."
            )
        assert self.model is not None and self.data is not None
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.is_connected = True

    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        if not self.is_connected or self.renderer is None:
            raise RobotDeviceNotConnectedError(
                f"MujocoCamera(camera_name='{self.config.camera_name}', camera_id={self.config.camera_id}) is not connected. Try running `camera.connect()` first."
            )
        
        if temporary_color_mode is not None:
            # TODO: Mujoco renderer always returns RGB. Add support for BGR if needed.
            raise NotImplementedError("temporary_color_mode is not supported for MujocoCamera yet.")

        start_time = time.perf_counter()
        # Assuming the renderer returns an RGB image by default
        self.renderer.update_scene(self.data, camera=self.config.camera_name)
        color_image = self.renderer.render()

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        self.logs["timestamp_utc"] = capture_timestamp_utc() # TODO: Decide if needed for sim

        self.color_image = color_image
        return color_image

    def async_read(self) -> np.ndarray:
        return self.read()

    # def read_loop(self):
    #     while not self.stop_event.is_set():
    #         try:
    #             self.color_image = self.read()
    #             # TODO: Add a small sleep if fps is a concern, though MuJoCo rendering might be tied to simulation steps
    #             # time.sleep(1 / self.fps) 
    #         except Exception as e:
    #             print(f"Error reading in thread: {e}")

    # def async_read(self) -> np.ndarray:
    #     if not self.is_connected:
    #         raise RobotDeviceNotConnectedError(
    #             f"MujocoCamera(camera_name='{self.config.camera_name}', camera_id={self.config.camera_id}) is not connected. Try running `camera.connect()` first."
    #         )

    #     if self.thread is None:
    #         self.stop_event = threading.Event()
    #         self.thread = threading.Thread(target=self.read_loop, args=())
    #         self.thread.daemon = True
    #         self.thread.start()

    #     # TODO: Add a timeout mechanism like in OpenCVCamera
    #     # num_tries = 0
    #     while True:
    #         if self.color_image is not None:
    #             return self.color_image
    #         time.sleep(0.01) # Small sleep to avoid busy waiting tight loop

    def disconnect(self):
        if not self.is_connected:
            # Allow disconnect to be called even if not connected (e.g. in __del__)
            return

        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()  # wait for the thread to finish
            self.thread = None
            self.stop_event = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect() 