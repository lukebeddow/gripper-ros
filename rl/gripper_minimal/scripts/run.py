#!/usr/bin/env python

import sys, signal, subprocess, time

timeout_before_kill = 1.0  # [s]
timeout_after_kill = 1.0   # [s]

# define the process names we want to kill
processes_to_kill = "gzclient gzserver roslaunch"

# now build up shell commands to kill these processes
template_one = "killall -q {}"
template_two = "killall -9 -q {}"
to_kill = processes_to_kill.split()
kill_one = ""
kill_two = ""
for i in range(len(to_kill)):
  kill_one += template_one.format(to_kill[i])
  kill_two += template_two.format(to_kill[i])
  if i != len(to_kill) - 1:
    kill_one += " & "
    kill_two += " & "

def signal_handler(sig, frame):

    print("killing processes now")

    global kill_one, kill_two

    time.sleep(timeout_before_kill)
    subprocess.call(kill_one, shell=True)
    time.sleep(timeout_after_kill)
    subprocess.call(kill_two, shell=True)
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    cmd = ' '.join(sys.argv[1:])

    cmd = "rosrun gripper_minimal gripper_env.py"

    print("command is:", cmd)

    subprocess.call(cmd, shell=True)