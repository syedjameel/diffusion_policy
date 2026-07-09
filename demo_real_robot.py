"""
Mello teleop for UR5e: move the arm with the Mello device and optionally record demos.

  python demo_real_robot.py -o <output_dir> --robot_ip <ur5e_ip>

See README_ur5e.md for UR5e and Mello setup. If the robot stalls,
tune gains with --osc_kp_pos and --osc_kp_rot. Keys: C=start record, S=stop, Q=quit,
Backspace=drop last episode. Use --debug for fixed joint positions (no Mello).
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import json
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from diffusion_policy.real_world.mello_teleop import MelloTeleopInterface, DummyMelloTeleopInterface
from diffusion_policy.real_world.ur10e_kinematics import (
    real_to_sim_joints, get_ee_pose, quat_to_axis_angle, apply_delta_pose,
)


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--mello_port', '-mp', default='/dev/serial/by-id/usb-M5Stack_Technology_Co.__Ltd_M5Stack_UiFlow_2.0_24587ce945900000-if00', help="Mello device serial port")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving command to executing on Robot in Sec.")
@click.option('--debug', is_flag=True, help="Use dummy Mello interface with fixed joint positions for testing.")
@click.option('--osc_kp_pos', default=1000.0, type=float, help="OSC position stiffness (default 1000)")
@click.option('--osc_kp_rot', default=50.0, type=float, help="OSC rotation stiffness (default 50)")
@click.option('--keyboard', is_flag=True, help="Keyboard Cartesian jog (no Mello needed): i/k=+-x, j/l=+-y, u/m=+-z, o/p=+-yaw, g=gripper toggle.")
def main(output, robot_ip, mello_port, vis_camera_idx, init_joints, frequency, command_latency, debug, osc_kp_pos, osc_kp_rot, keyboard):

    # 3x RealSense D405 -- no advanced-mode preset (415/435/455 JSONs are model-specific and
    # a D405 rejects them). D405s open with defaults; None skips the preset load.
    configs = None

    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        # --keyboard needs no Mello device; reuse the dummy as a placeholder.
        MelloInterface = DummyMelloTeleopInterface if (debug or keyboard) else MelloTeleopInterface
        mello_kwargs = {} if (debug or keyboard) else {'port': mello_port}
        with KeystrokeCounter() as key_counter, \
            MelloInterface(**mello_kwargs) as mello, \
            RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                obs_image_resolution=(640,480),
                # 3x D405 in front/side/wrist order (positional role mapping in real_env):
                # front=409122273078, side=323622272232, wrist=409122272284.
                camera_serial_numbers=['409122273078', '323622272232', '409122272284'],
                camera_configs=configs,
                frequency=frequency,
                init_joints=init_joints,
                # keyboard jog drives an absolute EE target -- the same Cartesian command
                # path eval_real_robot uses; Mello/debug keep joint mode.
                action_mode='cartesian' if keyboard else 'joint',
                enable_multi_cam_vis=True,
                record_raw_video=True,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                osc_kp_pos=osc_kp_pos,
                osc_kp_rot=osc_kp_rot,
            ) as env:
            cv2.setNumThreads(1)

            time.sleep(1.0)
            kd_pos = 2 * np.sqrt(osc_kp_pos) * 1.0
            kd_rot = 2 * np.sqrt(osc_kp_rot) * 1.0
            print(f'OSC: Kp_pos={osc_kp_pos}, Kp_rot={osc_kp_rot}, Kd_pos={kd_pos:.1f}, Kd_rot={kd_rot:.1f}')
            print('Ready!')
            # Keyboard jog state: absolute EE target seeded from the CURRENT pose
            # (FK of the sim-frame joints -- arm_joint_pos is already converted at the
            # RTDE boundary), then nudged per keypress. 1 cm / 5 deg per press.
            kb_pos = kb_quat = None
            kb_gripper = 1.0  # >0 = open
            # Small steps: key auto-repeat (~25 Hz held) then advances the target
            # ~7 cm/s CONTINUOUSLY -- sustained force through the stiction dead zone,
            # like the policy's 10 Hz stream. 1 cm discrete steps stick-slipped: a 1 cm
            # error is only ~10 N task force, below the sysid'd 20-30 N*m joint stiction,
            # so presses stacked then lurched.
            KB_STEP_POS = 0.003
            KB_STEP_ROT = np.deg2rad(2.0)
            KB_LEASH = 0.05  # clamp target within 5 cm of the actual EE (no error stacking)
            if keyboard:
                obs0 = env.get_obs()
                kb_pos, kb_quat = get_ee_pose(obs0['arm_joint_pos'][-1])
                print(f'[keyboard] jog from EE pos {np.round(kb_pos,3)} | '
                      f'i/k=+-x  j/l=+-y  u/m=+-z  o/p=+-yaw  g=gripper  q=quit')
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                obs = env.get_obs()

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Guard: only when there is an episode to drop. The click.confirm
                        # BLOCKS the teleop loop (the arm holds the last OSC target), and a
                        # stray backspace used to freeze/crash the session on an empty buffer.
                        if env.replay_buffer.n_episodes < 1:
                            print('[backspace ignored: no recorded episodes]')
                        elif click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                    elif keyboard and kb_pos is not None:
                        # Cartesian jog: build a 6D delta and apply to the held target.
                        jog = {
                            KeyCode(char='i'): ( KB_STEP_POS, 0, 0, 0, 0, 0),
                            KeyCode(char='k'): (-KB_STEP_POS, 0, 0, 0, 0, 0),
                            KeyCode(char='j'): (0,  KB_STEP_POS, 0, 0, 0, 0),
                            KeyCode(char='l'): (0, -KB_STEP_POS, 0, 0, 0, 0),
                            KeyCode(char='u'): (0, 0,  KB_STEP_POS, 0, 0, 0),
                            KeyCode(char='m'): (0, 0, -KB_STEP_POS, 0, 0, 0),
                            KeyCode(char='o'): (0, 0, 0, 0, 0,  KB_STEP_ROT),
                            KeyCode(char='p'): (0, 0, 0, 0, 0, -KB_STEP_ROT),
                        }.get(key_stroke)
                        if jog is not None:
                            kb_pos, kb_quat = apply_delta_pose(kb_pos, kb_quat, np.array(jog))
                            _cur = get_ee_pose(obs['arm_joint_pos'][-1])[0]
                            print(f'[keyboard] target {np.round(kb_pos,3)} | actual {np.round(_cur,3)}')
                        elif key_stroke == KeyCode(char='g'):
                            kb_gripper = -kb_gripper
                            print(f'[keyboard] gripper -> {"CLOSE" if kb_gripper < 0 else "OPEN"}')
                stage = key_counter[Key.space]

                # visualize -- our real_env names camera obs by ROLE (front/side/wrist_rgb,
                # positional order = the serial list), not camera_{i}
                vis_key = ('front_rgb', 'side_rgb', 'wrist_rgb')[vis_camera_idx]
                vis_img = obs[vis_key][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                if debug:
                    text += ' (DEBUG)'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )
                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)

                if keyboard:
                    # leash: PER-AXIS clamp of the target to within KB_LEASH of the actual
                    # EE. Per-axis is essential: the earlier radial clamp scaled the y/z
                    # error down along with x, so the target ADOPTED whatever lateral drift
                    # the arm picked up (stiction coupling) -- a drift integrator that
                    # walked y/z away (-9 cm y observed). Per-axis leaves untouched any
                    # axis whose error is < KB_LEASH (full Kp correction authority), and
                    # only caps the axis you are deliberately leading (no press stacking).
                    cur_pos, _ = get_ee_pose(obs['arm_joint_pos'][-1])
                    kb_pos = cur_pos + np.clip(kb_pos - cur_pos, -KB_LEASH, KB_LEASH)
                    # absolute EE target [pos, axis-angle] + gripper (cartesian mode)
                    unified_action = np.concatenate(
                        [kb_pos, quat_to_axis_angle(kb_quat), [kb_gripper]])
                else:
                    mello_values = mello.get_latest_values()
                    # Mello is a physical replica arm -> its joints are REAL pendant-frame.
                    # exec_actions joint targets are SIM-frame (the controller FKs them), so
                    # convert here at the device boundary (rig-orientation note in
                    # ur10e_kinematics; without this the OSC pulls the arm 90 deg off).
                    mello_joints = real_to_sim_joints(mello_values[:6])
                    gripper_command = mello_values[6]
                    unified_action = np.concatenate([mello_joints, [gripper_command]])

                env.exec_actions(
                    actions=[unified_action], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
