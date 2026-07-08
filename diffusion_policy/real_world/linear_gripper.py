"""Serial open/close driver for the custom linear parallel-jaw gripper.

Drop-in replacement for ``robotiq_gripper.RobotiqGripper`` in the OSC controller, adapted
from ``RC10_control/rc10_api/gripper.py``. The gripper firmware takes two ASCII commands over
a serial port: ``Open\\n`` and ``Close\\n``. There is no position/force API and no encoder
feedback -- the OmniReset stack only ever commands a boolean close state and observes the
*commanded* ``last_gripper_action`` (never jaw position), exactly as it did for the 2F-85.

Hardened for use inside the 500 Hz RTDE controller process:
  * writes only on state TRANSITIONS (no per-step serial traffic in the torque loop);
  * serial exceptions are swallowed with a warning -- a USB/serial hiccup must never raise
    out of the control loop and drop the arm;
  * the serial port is opened by the caller inside the child process's ``run()`` (never
    across a fork), matching how ``RobotiqGripper`` was used.
"""

from __future__ import annotations

import serial


class LinearGripper:
    def __init__(self, device: str = "/dev/ttyACM0", baudrate: int = 115200, timeout: float = 1.0):
        self._serial = serial.Serial(device, baudrate, timeout=timeout)
        self._device = device
        # None = unknown until the first command is sent (so the first set_closed always writes).
        self._is_open: bool | None = None

    def set_closed(self, closed: bool) -> None:
        """Command close (``True``) or open (``False``). Writes only on a state change."""
        want_open = not bool(closed)
        if want_open is self._is_open:
            return
        try:
            self._serial.write(b"Open\n" if want_open else b"Close\n")
            self._is_open = want_open
        except (serial.SerialException, OSError) as e:  # don't let a serial hiccup drop the arm
            print(f"[LinearGripper] serial write failed ({self._device}): {e}")

    def send(self, state) -> None:
        """rc10-compatible alias: ``state > 0`` opens, ``state <= 0`` closes."""
        self.set_closed(state <= 0)

    @property
    def is_open(self) -> bool | None:
        return self._is_open

    def disconnect(self) -> None:
        try:
            if self._serial.is_open:
                self._serial.close()
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass
