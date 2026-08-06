"""PS4 joystick for eval_real_robot: episode control + manual takeover jog.

Adapted from the proven lerobot-rig implementation
(RC10_control/rc10_api/ps4_joystick.py) -- same pygame backend and the same
button/axis indices verified on this rig's controller:
    buttons: 0=X/cross, 1=Circle, 2=Triangle, 3=Square, 4=L1, 5=R1
    axes:    0=LX, 1=LY, 2=L2, 3=RX, 4=RY, 5=R2  (Linux DS4 mapping)

Bindings consumed by eval_real_robot.py:
    hold R1                     -> takeover: policy paused, sticks jog the arm
    trigger (L2 or R2) + Triangle -> end episode marked SUCCESS, reset robot
    trigger (L2 or R2) + Circle   -> end episode marked FAILURE, reset robot
    X/cross                     -> toggle gripper open/closed (used in takeover)
    hold L1                     -> slow jog

The lerobot convention had NO trigger chord (plain Triangle/Circle fired
success/terminate); the chord here is deliberate, to make accidental resets
impossible while the controller is being handled.
"""

import threading
import time

import pygame


class PS4EvalJoystick:
    BTN_CROSS = 0
    BTN_CIRCLE = 1
    BTN_TRIANGLE = 2
    BTN_SQUARE = 3
    BTN_L1 = 4
    BTN_R1 = 5
    AX_LX = 0
    AX_LY = 1
    AX_L2 = 2
    AX_RX = 3
    AX_RY = 4
    AX_R2 = 5

    def __init__(self, deadzone=0.05, alpha=0.3, poll_rate=100,
                 slow_factor=0.2, trigger_threshold=0.5):
        self.deadzone = deadzone
        self.alpha = alpha
        self.dt = 1.0 / poll_rate
        self.slow_factor = slow_factor
        # trigger axes rest at -1.0 (and report 0.0 until first touched in SDL),
        # so > +0.5 is a safe "held" test in both states
        self.trigger_threshold = trigger_threshold

        # smoothed stick deflections in [-1, 1]
        self._sx = 0.0  # LX
        self._sy = 0.0  # LY
        self._sz = 0.0  # RY
        self._syaw = 0.0  # RX

        self._takeover = False
        self._slow = False
        self._gripper_state = 1.0  # +1 open (default), -1 closed; X toggles
        self._success_reset = False  # edge flags, cleared on read
        self._failure_reset = False

        self._prev_cross = False
        self._prev_triangle = False
        self._prev_circle = False

        self._lock = threading.Lock()

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            pygame.quit()
            raise RuntimeError("No PS4 controller found")
        self._controller = pygame.joystick.Joystick(0)
        self._controller.init()
        self.name = self._controller.get_name()

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return (value - (self.deadzone if value > 0 else -self.deadzone)) \
            / (1 - self.deadzone)

    def _poll_loop(self):
        while self._running:
            pygame.event.pump()

            lx = self._apply_deadzone(self._controller.get_axis(self.AX_LX))
            ly = self._apply_deadzone(self._controller.get_axis(self.AX_LY))
            ry = self._apply_deadzone(self._controller.get_axis(self.AX_RY))
            rx = self._apply_deadzone(self._controller.get_axis(self.AX_RX))

            trigger_held = (
                self._controller.get_axis(self.AX_L2) > self.trigger_threshold
                or self._controller.get_axis(self.AX_R2) > self.trigger_threshold)
            cross = bool(self._controller.get_button(self.BTN_CROSS))
            triangle = bool(self._controller.get_button(self.BTN_TRIANGLE))
            circle = bool(self._controller.get_button(self.BTN_CIRCLE))
            takeover = bool(self._controller.get_button(self.BTN_R1))
            slow = bool(self._controller.get_button(self.BTN_L1))

            with self._lock:
                a = self.alpha
                self._sx = a * lx + (1 - a) * self._sx
                self._sy = a * ly + (1 - a) * self._sy
                self._sz = a * ry + (1 - a) * self._sz
                self._syaw = a * rx + (1 - a) * self._syaw

                self._takeover = takeover
                self._slow = slow
                if cross and not self._prev_cross:
                    self._gripper_state = -self._gripper_state
                if trigger_held and triangle and not self._prev_triangle:
                    self._success_reset = True
                if trigger_held and circle and not self._prev_circle:
                    self._failure_reset = True

            self._prev_cross = cross
            self._prev_triangle = triangle
            self._prev_circle = circle

            time.sleep(self.dt)

    # ========= consumer API =========
    def is_takeover(self):
        """True while R1 is held: policy paused, sticks control the arm."""
        with self._lock:
            return self._takeover

    def get_jog(self):
        """Smoothed jog command (jx, jy, jz, jyaw), each in [-1, 1].

        Frame: REP-103 base (x forward, y left, z up).
        left stick up = +x, left stick left = +y,
        right stick up = +z, right stick left = +yaw.
        L1 held scales everything by slow_factor.
        """
        with self._lock:
            scale = self.slow_factor if self._slow else 1.0
            return (-self._sy * scale, -self._sx * scale,
                    -self._sz * scale, -self._syaw * scale)

    def get_gripper_state(self):
        """+1.0 open / -1.0 closed; toggled by X/cross."""
        with self._lock:
            return self._gripper_state

    def set_gripper_state(self, state):
        """Force the X-toggle latch (+1.0 open / -1.0 closed). Used to re-sync
        after scripted gripper moves (e.g. close-at-home on reset), so entering
        takeover afterwards doesn't instantly command a stale state."""
        with self._lock:
            self._gripper_state = float(state)

    def get_reset_events(self):
        """Edge-triggered chord events, cleared on read:
        {'success': trigger+Triangle pressed, 'failure': trigger+Circle pressed}
        """
        with self._lock:
            events = {'success': self._success_reset,
                      'failure': self._failure_reset}
            self._success_reset = False
            self._failure_reset = False
            return events

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        pygame.quit()
