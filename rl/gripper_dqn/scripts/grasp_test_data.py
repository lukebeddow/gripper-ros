import numpy as np
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

global palm_force_threshold, X_force_threshold, Y_force_threshold
palm_force_threshold = 5
X_force_threshold = 5
Y_force_threshold = 5

def set_palm_frc_threshold(value):
  global palm_force_threshold
  palm_force_threshold = value

def set_X_frc_threshold(value):
  global X_force_threshold
  X_force_threshold = value

def set_Y_frc_threshold(value):
  global Y_force_threshold
  Y_force_threshold = value

class GraspTestData:

  # data structures for saving testing data
  step_data = namedtuple("step_data", ("step_num", "state_vector", "SI_state", "action"))
  image_data = namedtuple("image_data", ("step_num", "rgb", "depth"))

  # define which shapes are which in the object set
  sphere_min = 1
  sphere_max = 6
  cuboid_min = 7
  cuboid_max = 15
  cylinder_min = 16
  cylinder_max = 27
  cube_min = 28
  cube_max = 30

  @dataclass
  class TrialData:
    object_name: str
    object_num: int
    trial_num: int
    steps: list
    images: list
    stable_height: bool
    target_height: bool
    lifted: bool
    exceed_bending: bool
    exceed_axial: bool
    exceed_limits: bool
    loop: bool
    dropped: bool
    out_of_bounds: bool
    exceed_palm: bool
    palm_frc_tol: float
    palm_tol_grasp: bool
    X_frc_tol: float
    Y_frc_tol: float
    X_tol_grasp: bool
    Y_tol_grasp: bool
    finger1_force: float
    finger2_force: float
    finger3_force: float
    palm_force: float
    info: str

  @dataclass
  class TestData:
    trials: list
    test_name: str
    finger_width: float
    finger_thickness: float
    heuristic: bool
    bend_gauge: bool
    palm_sensor: bool
    wrist_Z_sensor: bool
    group_name: str
    run_name: str
    best_SR: float
    best_EP: float

  @dataclass
  class TestResults:
    num_trials: int = 0
    num_objects: int = 0
    num_sphere: int = 0
    num_cuboid: int = 0
    num_cylinder: int = 0
    num_cube: int = 0
    avg_obj_num_trials: float = 0
    success_rate: float = 0
    avg_obj_success_rate: float = 0
    sphere_SR: float = 0
    cylinder_SR: float = 0
    cuboid_SR: float = 0
    cube_SR: float = 0
    avg_steps: float = 0
    avg_stable_height: float = 0
    avg_target_height: float = 0
    avg_lifted: float = 0
    avg_exceed_bending: float = 0
    avg_exceed_axial: float = 0
    avg_exceed_limits: float = 0
    avg_loop: float = 0
    avg_dropped: float = 0
    avg_out_of_bounds: float = 0
    avg_exceed_palm: float = 0
    avg_palm_frc_tol: float = 0
    avg_palm_frc_under: float = 0
    avg_palm_frc_saturated: float = 0
    num_palm_frc_tol: int = 0
    avg_X_frc_tol: float = 0
    avg_X_frc_under: float = 0
    avg_X_frc_saturated: float = 0
    num_X_frc_tol: int = 0
    avg_Y_frc_tol: float = 0
    avg_Y_frc_under: float = 0
    avg_Y_frc_saturated: float = 0
    num_Y_frc_tol: int = 0
    num_X_probe: int = 0
    num_Y_probe: int = 0
    num_Z_probe: int = 0
    avg_f1_frc: float = 0
    avg_f2_frc: float = 0
    avg_f3_frc: float = 0
    avg_f_frc: float = 0
    avg_p_frc: float = 0
    avg_SR_per_obj: float = 0

  def __init__(self):
    """
    Empty initialisation, should now call start_test(...)
    """

    self.add_palm_start_force = False
    pass

  def capture_depth_image(self):
    """
    This should be overloaded with a viable function
    """
    raise NotImplementedError

  def start_test(self, test_name, rl_obj, depth_image_fcn, image_rate=1, heuristic=False):
    """
    Data for an entire test
    """

    best_sr, best_ep = rl_obj.trainer.calc_best_performance()

    # create test data structure
    self.data = GraspTestData.TestData(
      [],                                       # trials
      test_name,                                # test_name
      rl_obj.trainer.env.params.finger_width,          # finger width
      rl_obj.trainer.env.params.finger_thickness,      # finger thickness
      heuristic,                                # is heuristic test
      rl_obj.trainer.env.mj.set.bending_gauge.in_use,  # bending sensor
      rl_obj.trainer.env.mj.set.palm_sensor.in_use,    # palm sensor
      rl_obj.trainer.env.mj.set.wrist_sensor_Z.in_use, # wrist sensor
      rl_obj.group_name,                       # run group name
      rl_obj.run_name,                         # run name
      best_sr,                                  # best test success rate in sim
      best_ep,                                  # episode at best_sr
    )

    self.image_data = GraspTestData.TestData(
      [],                                       # trials
      test_name,                                # test_name
      rl_obj.trainer.env.params.finger_width,          # finger width
      rl_obj.trainer.env.params.finger_thickness,      # finger thickness
      heuristic,                                # is heuristic test
      rl_obj.trainer.env.mj.set.bending_gauge.in_use,  # bending sensor
      rl_obj.trainer.env.mj.set.palm_sensor.in_use,    # palm sensor
      rl_obj.trainer.env.mj.set.wrist_sensor_Z.in_use, # wrist sensor
      rl_obj.group_name,                       # run group name
      rl_obj.run_name,                         # run name
      best_sr,                                  # best test success rate in sim
      best_ep,                                  # episode at best_sr
    )
    
    # initialise class variables
    self.image_rate = image_rate
    self.current_trial = None
    self.current_trial_with_images = None

    # assign depth function pointer to this class
    self.capture_depth_image = depth_image_fcn

  def start_trial(self, object_name, object_num, trial_num):
    """
    Begin a new trial
    """

    self.current_trial = GraspTestData.TrialData(
      object_name,    # object_name
      object_num,     # object_num
      trial_num,      # trial_num
      [],             # steps
      [],             # images
      0,              # stable_height
      0,              # target_height
      0,              # lifted
      0,              # exceed_bending
      0,              # exceed_axial
      0,              # exceed_limits
      0,              # loop
      0,              # dropped
      0,              # out_of_bounds
      0,              # exceed_palm
      0,              # palm frc tol
      0,              # palm_tol_grasp
      0,              # X_frc_tol
      0,              # Y_frc_tol
      0,              # X_tol_grasp
      0,              # Y_tol_grasp
      0,              # finger1_force
      0,              # finger2_force
      0,              # finger3_force
      0,              # palm force
      "",             # info string
    )
    self.current_trial_with_images = GraspTestData.TrialData(
      object_name,    # object_name
      object_num,     # object_num
      trial_num,      # trial_num
      [],             # steps
      [],             # images
      0,              # stable_height
      0,              # target_height
      0,              # lifted
      0,              # exceed_bending
      0,              # exceed_axial
      0,              # exceed_limits
      0,              # loop
      0,              # dropped
      0,              # out_of_bounds
      0,              # exceed_palm
      0,              # palm frc tol
      0,              # palm_tol_grasp
      0,              # X_frc_tol
      0,              # Y_frc_tol
      0,              # X_tol_grasp
      0,              # Y_tol_grasp
      0,              # finger1_force
      0,              # finger2_force
      0,              # finger3_force
      0,              # palm force
      "",             # info string
    )

    self.current_step_count = 0

  def add_step(self, state_vector, action, SI_vector=None):
    """
    Add data for a single step
    """

    if self.current_trial == None:
      raise RuntimeError("current_trial is None in GraspTestData class")

    self.current_step_count += 1

    # add step data to the current trial
    this_step = GraspTestData.step_data(
      self.current_step_count,    # step_num
      state_vector,               # state_vector
      SI_vector,                  # SI_state (state vector calibrated but not normalised)
      action                      # action
    )
    self.current_trial.steps.append(this_step)

    # are we taking a photo this step
    if (self.current_step_count - 1) % self.image_rate == 0:
      rgb, depth = self.capture_depth_image()
      this_image = GraspTestData.image_data(
        self.current_step_count,  # step_num
        rgb,                      # rgb
        depth                     # depth
      )
      # add step image pairs
      self.current_trial_with_images.steps.append(this_step)
      self.current_trial_with_images.images.append(this_image)

  def finish_trial(self, request, stable_forces=None):
    """
    Finish a trial and save the results
    """

    if stable_forces is None:
      if request.s_h:
        raise RuntimeError("stable height is TRUE but there are NO stable grasp forces")
      stable_forces = [0, 0, 0, 0]
    elif len(stable_forces) != 4:
      raise RuntimeError(f"Stable forces of {stable_forces}, invalid input")
    else:
      print("Trial stable forces are:", stable_forces)

    # special cases where one flag implies another
    if request.s_h: request.t_h = 1 # ie True
    if request.t_h: request.lft = 1
    if request.drp: request.lft = 1

    # if not request.s_h: request.pFTol = 0.0

    self.current_trial.stable_height = request.s_h
    self.current_trial.target_height = request.t_h
    self.current_trial.lifted = request.lft
    self.current_trial.exceed_bending = request.xBnd
    self.current_trial.exceed_axial = request.xAxl
    self.current_trial.exceed_limits = request.xLim
    self.current_trial.exceed_palm = request.xPlm
    self.current_trial.loop = request.loop
    self.current_trial.dropped = request.drp
    self.current_trial.out_of_bounds = request.oob
    self.current_trial.palm_frc_tol = request.pFTol
    self.current_trial.X_frc_tol = request.XFTol
    self.current_trial.Y_frc_tol = request.YFTol
    self.current_trial.info = request.info
    self.current_trial.finger1_force = stable_forces[0]
    self.current_trial.finger2_force = stable_forces[1]
    self.current_trial.finger3_force = stable_forces[2]
    self.current_trial.palm_force = stable_forces[3]
    self.data.trials.append(deepcopy(self.current_trial))
    self.current_trial = None

    self.current_trial_with_images.stable_height = request.s_h
    self.current_trial_with_images.target_height = request.t_h
    self.current_trial_with_images.lifted = request.lft
    self.current_trial_with_images.exceed_bending = request.xBnd
    self.current_trial_with_images.exceed_axial = request.xAxl
    self.current_trial_with_images.exceed_limits = request.xLim
    self.current_trial_with_images.exceed_palm = request.xPlm
    self.current_trial_with_images.loop = request.loop
    self.current_trial_with_images.dropped = request.drp
    self.current_trial_with_images.out_of_bounds = request.oob
    self.current_trial_with_images.palm_frc_tol = request.pFTol
    self.current_trial_with_images.X_frc_tol = request.XFTol
    self.current_trial_with_images.Y_frc_tol = request.YFTol
    self.current_trial_with_images.info = request.info
    self.current_trial_with_images.finger1_force = stable_forces[0]
    self.current_trial_with_images.finger2_force = stable_forces[1]
    self.current_trial_with_images.finger3_force = stable_forces[2]
    self.current_trial_with_images.palm_force = stable_forces[3]
    self.image_data.trials.append(deepcopy(self.current_trial_with_images))
    self.current_trial_with_images = None

  def get_test_results(self, print_trials=False, data=None):
    """
    Get data structure of test information
    """

    if data is None: data = self.data

    entries = []
    object_nums = []
    num_trials = 0

    num_palm_frc_under_threshold = 0
    cum_palm_frc_under_threshold = 0
    cum_palm_frc_saturated = 0

    num_X_probe = 0
    num_Y_probe = 0
    num_Z_probe = 0

    num_X_frc_under_threshold = 0
    cum_X_frc_under_threshold = 0
    cum_X_frc_saturated = 0

    num_Y_frc_under_threshold = 0
    cum_Y_frc_under_threshold = 0
    cum_Y_frc_saturated = 0

    if len(data.trials) == 0:
      print("WARNING: get_test_results() found 0 trials")
      return None

    # sort trial data
    for trial in data.trials:

      # ignore any object number over 100
      if trial.object_num > 100: continue

      num_trials += 1

      if trial.palm_frc_tol > 0.01 and trial.stable_height:
        if self.add_palm_start_force:
          trial.palm_frc_tol += trial.palm_force
        num_Z_probe += 1
        if trial.palm_frc_tol < palm_force_threshold - 1e-5:
          num_palm_frc_under_threshold += 1
          cum_palm_frc_under_threshold += trial.palm_frc_tol
          cum_palm_frc_saturated += trial.palm_frc_tol
        elif trial.palm_frc_tol > palm_force_threshold - 1e-5:
          cum_palm_frc_saturated += palm_force_threshold

      if trial.X_frc_tol > 0.01 and trial.stable_height:
        num_X_probe += 1
        if trial.X_frc_tol < X_force_threshold - 1e-5:
          num_X_frc_under_threshold += 1
          cum_X_frc_under_threshold += trial.X_frc_tol
          cum_X_frc_saturated += trial.X_frc_tol
        elif trial.X_frc_tol > X_force_threshold - 1e-5:
          cum_X_frc_saturated += X_force_threshold

      if trial.Y_frc_tol > 0.01 and trial.stable_height:
        num_Y_probe += 1
        if trial.Y_frc_tol < Y_force_threshold - 1e-5:
          num_Y_frc_under_threshold += 1
          cum_Y_frc_under_threshold += trial.Y_frc_tol
          cum_Y_frc_saturated += trial.Y_frc_tol
        elif trial.Y_frc_tol > Y_force_threshold - 1e-5:
          cum_Y_frc_saturated += Y_force_threshold
            
      found = False
      for j in range(len(object_nums)):
        if object_nums[j] == trial.object_num:
          found = True
          break

      if not found:

        # create a new entry for this object
        new_entry = deepcopy(trial)
        new_entry.trial_num = 1
        new_entry.palm_frc_tol *= trial.stable_height # set to zero if no stable height
        new_entry.palm_tol_grasp += (trial.palm_frc_tol > palm_force_threshold - 1e-5) * trial.stable_height
        new_entry.X_frc_tol *= trial.stable_height # set to zero if no stable height
        new_entry.X_tol_grasp += (trial.X_frc_tol > X_force_threshold - 1e-5) * trial.stable_height
        new_entry.Y_frc_tol *= trial.stable_height # set to zero if no stable height
        new_entry.Y_tol_grasp += (trial.Y_frc_tol > Y_force_threshold - 1e-5) * trial.stable_height
        new_entry.finger1_force *= trial.stable_height # set to zero if no stable height
        new_entry.finger2_force *= trial.stable_height # set to zero if no stable height
        new_entry.finger3_force *= trial.stable_height # set to zero if no stable height
        new_entry.palm_force *= trial.stable_height # set to zero if no stable height
        entries.append(new_entry)
        object_nums.append(trial.object_num)

      else:

        # add to the existing entry for this object
        entries[j].object_name += trial.object_name
        entries[j].trial_num += 1
        entries[j].steps += trial.steps
        entries[j].stable_height += trial.stable_height
        entries[j].target_height += trial.target_height
        entries[j].lifted += trial.lifted
        entries[j].exceed_bending += trial.exceed_bending
        entries[j].exceed_axial += trial.exceed_axial
        entries[j].exceed_limits += trial.exceed_limits
        entries[j].loop += trial.loop
        entries[j].dropped += trial.dropped
        entries[j].out_of_bounds += trial.out_of_bounds
        entries[j].exceed_palm += trial.exceed_palm
        entries[j].palm_frc_tol += trial.palm_frc_tol * trial.stable_height
        entries[j].palm_tol_grasp += (trial.palm_frc_tol > palm_force_threshold - 1e-5) * trial.stable_height
        entries[j].X_frc_tol += trial.X_frc_tol * trial.stable_height
        entries[j].X_tol_grasp += (trial.X_frc_tol > X_force_threshold - 1e-5) * trial.stable_height
        entries[j].Y_frc_tol += trial.Y_frc_tol * trial.stable_height
        entries[j].Y_tol_grasp += (trial.Y_frc_tol > Y_force_threshold - 1e-5) * trial.stable_height
        entries[j].finger1_force += trial.finger1_force * trial.stable_height
        entries[j].finger2_force += trial.finger2_force * trial.stable_height
        entries[j].finger3_force += trial.finger3_force * trial.stable_height
        entries[j].palm_force += trial.palm_force * trial.stable_height
        entries[j].info += trial.info

    # create TestResults to return
    result = GraspTestData.TestResults()

    # now process trial data
    for i in range(len(entries)):

      if print_trials:
        print(f"Object num = {entries[i].object_num}, num trials = {entries[i].trial_num}, SH = {entries[i].stable_height}")

      result.num_trials += entries[i].trial_num
      result.num_objects += 1

      result.avg_steps += len(entries[i].steps)
      result.avg_stable_height += entries[i].stable_height
      result.avg_target_height += entries[i].target_height
      result.avg_lifted += entries[i].lifted
      result.avg_exceed_bending += entries[i].exceed_bending
      result.avg_exceed_axial += entries[i].exceed_axial
      result.avg_exceed_limits += entries[i].exceed_limits
      result.avg_loop += entries[i].loop
      result.avg_dropped += entries[i].dropped
      result.avg_out_of_bounds += entries[i].out_of_bounds
      result.avg_exceed_palm += entries[i].exceed_palm
      result.avg_palm_frc_tol += entries[i].palm_frc_tol
      result.num_palm_frc_tol += entries[i].palm_tol_grasp
      result.avg_X_frc_tol += entries[i].X_frc_tol
      result.num_X_frc_tol += entries[i].X_tol_grasp
      result.avg_Y_frc_tol += entries[i].Y_frc_tol
      result.num_Y_frc_tol += entries[i].Y_tol_grasp
      result.avg_f1_frc += entries[i].finger1_force
      result.avg_f2_frc += entries[i].finger2_force
      result.avg_f3_frc += entries[i].finger3_force
      result.avg_p_frc += entries[i].palm_force
      if entries[i].stable_height > 0:
        result.avg_SR_per_obj += float(entries[i].stable_height) / float(entries[i].trial_num)

      if (entries[i].object_num >= self.sphere_min and
          entries[i].object_num <= self.sphere_max):
        result.num_sphere += entries[i].trial_num
        result.sphere_SR += entries[i].stable_height

      if (entries[i].object_num >= self.cuboid_min and
          entries[i].object_num <= self.cuboid_max):
        result.num_cuboid += entries[i].trial_num
        result.cuboid_SR += entries[i].stable_height


      if (entries[i].object_num >= self.cube_min and
          entries[i].object_num <= self.cube_max):
        result.num_cube += entries[i].trial_num
        result.cube_SR += entries[i].stable_height


      if (entries[i].object_num >= self.cylinder_min and
          entries[i].object_num <= self.cylinder_max):
        result.num_cylinder += entries[i].trial_num
        result.cylinder_SR += entries[i].stable_height


    # finalise and normalise
    num_stable_height = result.avg_stable_height # store this before it is averaged
    result.num_X_probe = num_X_probe
    result.num_Y_probe = num_Y_probe
    result.num_Z_probe = num_Z_probe
    result.avg_steps /= result.num_trials
    result.avg_stable_height /= result.num_trials
    result.avg_target_height /= result.num_trials
    result.avg_lifted /= result.num_trials
    result.avg_exceed_bending /= result.num_trials
    result.avg_exceed_axial /= result.num_trials
    result.avg_exceed_limits /= result.num_trials
    result.avg_loop /= result.num_trials
    result.avg_dropped /= result.num_trials
    result.avg_out_of_bounds /= result.num_trials
    result.avg_exceed_palm /= result.num_trials
    result.avg_SR_per_obj /= result.num_objects

    # determine palm force tolerances
    if result.num_Z_probe > 0:
      result.avg_palm_frc_tol /= result.num_Z_probe
      if num_palm_frc_under_threshold == 0:
        result.avg_palm_frc_under = 0.0
      else:
        result.avg_palm_frc_under = cum_palm_frc_under_threshold / num_palm_frc_under_threshold
      result.avg_palm_frc_saturated = cum_palm_frc_saturated / result.num_Z_probe
      result.num_palm_frc_tol /= result.num_Z_probe

    # determine X force tolerances
    if result.num_X_probe > 0:
      result.avg_X_frc_tol /= result.num_X_probe
      if num_X_frc_under_threshold == 0:
        result.avg_X_frc_under = 0.0
      else:
        result.avg_X_frc_under = cum_X_frc_under_threshold / num_X_frc_under_threshold
      result.avg_X_frc_saturated = cum_X_frc_saturated / result.num_X_probe
      result.num_X_frc_tol /= result.num_X_probe

    # determine Y force tolerances
    if result.num_Y_probe > 0:
      result.avg_Y_frc_tol /= result.num_Y_probe
      if num_Y_frc_under_threshold == 0:
        result.avg_Y_frc_under = 0.0
      else:
        result.avg_Y_frc_under = cum_Y_frc_under_threshold / num_Y_frc_under_threshold
      result.avg_Y_frc_saturated = cum_Y_frc_saturated / result.num_Y_probe
      result.num_Y_frc_tol /= result.num_Y_probe

    # get average forces from the end of stable grasps
    if num_stable_height > 0:
      result.avg_f1_frc /= num_stable_height
      result.avg_f2_frc /= num_stable_height
      result.avg_f3_frc /= num_stable_height
      result.avg_p_frc /= num_stable_height
      result.avg_f_frc = (1.0/3.0) * (result.avg_f1_frc + result.avg_f2_frc + result.avg_f3_frc)

    if result.num_sphere > 0:
      result.sphere_SR /= result.num_sphere
    else:
      result.sphere_SR = 0

    if result.num_cuboid > 0:
      result.cuboid_SR /= result.num_cuboid
    else:
      result.cuboid_SR = 0

    if result.num_cylinder > 0:
      result.cylinder_SR /= result.num_cylinder
    else:
      result.cylinder_SR = 0
      
    if result.num_cube > 0:
      result.cube_SR /= result.num_cube
    else:
      result.cube_SR = 0

    return result

  def get_test_string(self, data=None, print_trials=False, detailed=False):
    """
    Print out information about the current test
    """

    # by default use any current test data
    if data is None: data = self.data

    info_str = """"""

    info_str += f"Test information\n\n"
    info_str += f"Test name: {data.test_name}\n"
    info_str += f"Finger width: {data.finger_width}\n"
    info_str += f"Finger thickness: {data.finger_thickness:.4f}\n"
    info_str += f"heuristic test: {data.heuristic}\n"
    info_str += f"Bending gauge in use: {data.bend_gauge}\n"
    info_str += f"Palm sensor in use: {data.palm_sensor}\n"
    info_str += f"Wrist Z sensor in use: {data.wrist_Z_sensor}\n"
    info_str += f"Loaded group name: {data.group_name}\n"
    info_str += f"Loaded run name: {data.run_name}\n"
    info_str += f"Loaded best SR: {data.best_SR:.3f}\n"

    results = self.get_test_results(data=data, print_trials=print_trials)

    if results is None: 
      info_str += "\nNO TRIAL DATA FOUND FOR THIS TEST"
      return info_str

    info_str += f"\nResults information:\n\n"

    if detailed:
      info_str += f"num_sphere = {results.num_sphere}\n"
      info_str += f"num_cuboid = {results.num_cuboid}\n"
      info_str += f"num_cylinder = {results.num_cylinder}\n"
      info_str += f"num_cube = {results.num_cube}\n"
      info_str += f"sphere_SR = {results.sphere_SR:.4f}\n"
      info_str += f"cylinder_SR = {results.cylinder_SR:.4f}\n"
      info_str += f"cuboid_SR = {results.cuboid_SR:.4f}\n"
      info_str += f"cube_SR = {results.cube_SR:.4f}\n"
      info_str += f"avg_steps = {results.avg_steps:.4f}\n"
      info_str += f"avg_stable_height = {results.avg_stable_height:.4f}\n"
      info_str += f"avg_target_height = {results.avg_target_height:.4f}\n"
      info_str += f"avg_lifted = {results.avg_lifted:.4f}\n"
      info_str += f"avg_exceed_bending = {results.avg_exceed_bending:.4f}\n"
      info_str += f"avg_exceed_axial = {results.avg_exceed_axial:.4f}\n"
      info_str += f"avg_exceed_limits = {results.avg_exceed_limits:.4f}\n"
      info_str += f"avg_loop = {results.avg_loop:.4f}\n"
      info_str += f"avg_dropped = {results.avg_dropped:.4f}\n"
      info_str += f"avg_out_of_bounds = {results.avg_out_of_bounds:.4f}\n"
      info_str += f"avg_exceed_palm = {results.avg_exceed_palm:.4f}\n"
      info_str += f"avg_SR_per_obj = {results.avg_SR_per_obj:.4f}\n"
      info_str += "\n"
      info_str += f"num_palm_probes = {results.num_Z_probe} out of {results.avg_stable_height * results.num_trials:.0f} (s.h)\n"
      info_str += f"avg_palm_frc_tol ({palm_force_threshold:.1f}N) = {results.avg_palm_frc_tol:.4f}\n"
      info_str += f"avg_palm_frc_under ({palm_force_threshold:.1f}N) = {results.avg_palm_frc_under:.4f}\n"
      info_str += f"avg_palm_frc_saturated ({palm_force_threshold:.1f}N) = {results.avg_palm_frc_saturated:.4f}\n"
      info_str += f"num_palm_frc_tol ({palm_force_threshold:.1f}N) = {results.num_palm_frc_tol:.4f}\n"
      info_str += "\n"
      info_str += f"num_X_probes = {results.num_X_probe} out of {results.avg_stable_height * results.num_trials:.0f} (s.h)\n"
      info_str += f"avg_X_frc_tol ({X_force_threshold:.1f}N) = {results.avg_X_frc_tol:.4f}\n"
      info_str += f"avg_X_frc_under ({X_force_threshold:.1f}N) = {results.avg_X_frc_under:.4f}\n"
      info_str += f"avg_X_frc_saturated ({X_force_threshold:.1f}N) = {results.avg_X_frc_saturated:.4f}\n"
      info_str += f"num_X_frc_tol ({X_force_threshold:.1f}N) = {results.num_X_frc_tol:.4f}\n"
      info_str += "\n"
      info_str += f"num_Y_probes = {results.num_Y_probe} out of {results.avg_stable_height * results.num_trials:.0f} (s.h)\n"
      info_str += f"avg_Y_frc_tol ({Y_force_threshold:.1f}N) = {results.avg_Y_frc_tol:.4f}\n"
      info_str += f"avg_Y_frc_under ({Y_force_threshold:.1f}N) = {results.avg_Y_frc_under:.4f}\n"
      info_str += f"avg_Y_frc_saturated ({Y_force_threshold:.1f}N) = {results.avg_Y_frc_saturated:.4f}\n"
      info_str += f"num_Y_frc_tol ({Y_force_threshold:.1f}N) = {results.num_Y_frc_tol:.4f}\n"
      info_str += "\n"
      info_str += f"average stable grasp finger1 force = {results.avg_f1_frc:.3f}N\n"
      info_str += f"average stable grasp finger2 force = {results.avg_f2_frc:.3f}N\n"
      info_str += f"average stable grasp finger3 force = {results.avg_f3_frc:.3f}N\n"
      info_str += f"average stable grasp palm force = {results.avg_p_frc:.3f}N\n"
      info_str += "\n"
      
    info_str += f"Sphere success rate: {results.sphere_SR:.4f}\n"
    info_str += f"cylinder success rate: {results.cylinder_SR:.4f}\n"
    info_str += f"cuboid success rate: {results.cuboid_SR:.4f}\n"
    info_str += f"cube success rate: {results.cube_SR:.4f}\n"
    info_str += "\n --- Key information ---\n\n"
    info_str += f"Total number of trials: {results.num_trials}\n"
    info_str += f"Total number of objects: {results.num_objects}\n"
    info_str += f"Grasp palm {palm_force_threshold:.1f}N stable % (out of successful grasps): {results.num_palm_frc_tol / results.avg_stable_height:.4f}\n"
    info_str += f"Grasp X {X_force_threshold:.1f}N stable % (out of successful grasps): {results.num_X_frc_tol / results.avg_stable_height:.4f}\n"
    info_str += f"Grasp Y {Y_force_threshold:.1f}N stable % (out of successful grasps): {results.num_Y_frc_tol / results.avg_stable_height:.4f}\n"
    info_str += f"Successful grasp average forces, finger = {results.avg_f_frc:.2f}N and palm = {results.avg_p_frc:.2f}N\n"
    info_str += "\n"
    info_str += f"Overall success rate: {results.avg_stable_height:.4f}\n"
    return info_str

  def print_trial(self, trial, steps=[0], state=True, sensor_state=[1, 5], SI=True, action=True):
    """
    Prints a given trial
    """

    from itertools import chain

    if state:
      print("Printing trial.state_vector data for the given steps:")
      col_format = ":<8"
      float_format = ".4f"
      top_str = "{0" + col_format[:] + "} "
      SI_str = "{0" + col_format[:] + "} "
      for x in range(len(trial.steps[0].state_vector)):
        top_str += "{" + str(x + 1) + col_format + "} "
        SI_str += "{" + str(x + 1) + col_format + float_format + "} "

      print(*("step", *[j for i in [("a", "b") for i in range(sensor_state[0])] for j in i]))

      print(top_str.format(
        "step", 
        *("bend1-1", *[j for i in [("diff", "bend1-" + str(i+2)) for i in range(sensor_state[0])] for j in i]), 
        *("bend2-1", *[j for i in [("diff", "bend2-" + str(i+2)) for i in range(sensor_state[0])] for j in i]), 
        *("bend3-1", *[j for i in [("diff", "bend3-" + str(i+2)) for i in range(sensor_state[0])] for j in i]), 
        *("palm-1", *[j for i in [("diff", "palm-" + str(i+2)) for i in range(sensor_state[0])] for j in i]), 
        *("wrist-1", *[j for i in [("diff", "wrist-" + str(i+2)) for i in range(sensor_state[0])] for j in i]), 
        *("x-1", *[j for i in [("diff", "x-" + str(i+2)) for i in range(sensor_state[1])] for j in i]), 
        *("y-1", *[j for i in [("diff", "y-" + str(i+2)) for i in range(sensor_state[1])] for j in i]), 
        *("z-1", *[j for i in [("diff", "z-" + str(i+2)) for i in range(sensor_state[1])] for j in i]), 
        *("H-1", *[j for i in [("diff", "H-" + str(i+2)) for i in range(sensor_state[1])] for j in i]), 
      )) 
      for i in steps:
        print(SI_str.format(i, *trial.steps[i].state_vector))

    if SI:
      print("Printing trial.SI_state data for the given steps:")
      col_format = ":<8"
      float_format = ".4f"
      top_str = "{0" + col_format[:] + "} "
      SI_str = "{0" + col_format[:] + "} "
      for x in range(len(trial.steps[0].SI_state)):
        top_str += "{" + str(x + 1) + col_format + "} "
        SI_str += "{" + str(x + 1) + col_format + float_format + "} "
      print(top_str.format("step", "bend1", "bend2", "bend3", "palm", "wrist", "x", "y", "z", "H"))
      for i in steps:
        print(SI_str.format(i, *trial.steps[i].SI_state))
      
