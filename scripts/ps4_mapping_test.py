"""Print live PS4 controller axes/buttons to verify the mapping on this machine.

Run, then move ONE control at a time and note which index reacts:
    python scripts/ps4_mapping_test.py

Expected (rig-verified, RC10_control convention -- what ps4_joystick.py assumes):
    axes:    0=LX  1=LY  2=L2  3=RX  4=RY  5=R2
    buttons: 0=X   1=Circle  2=Triangle  3=Square  4=L1  5=R1

If your machine's pygame/SDL reports a different order, update the AX_*/BTN_*
constants in diffusion_policy/real_world/ps4_joystick.py accordingly.
"""

import time

import pygame

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise SystemExit("No controller found")
js = pygame.joystick.Joystick(0)
js.init()
print(f"Controller: {js.get_name()}  "
      f"axes={js.get_numaxes()} buttons={js.get_numbuttons()}")
print("Move one stick/trigger or press one button at a time. Ctrl+C to exit.")

try:
    while True:
        pygame.event.pump()
        axes = ' '.join(f'a{i}:{js.get_axis(i):+.2f}'
                        for i in range(js.get_numaxes()))
        btns = ' '.join(f'b{i}:{js.get_button(i)}'
                        for i in range(js.get_numbuttons()))
        print(f'\r{axes} | {btns}   ', end='', flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print()
