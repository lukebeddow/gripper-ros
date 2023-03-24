import numpy as np
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

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

  def __init__(self):
    """
    Empty initialisation, should now call start_test(...)
    """

    pass

  def capture_depth_image(self):
    """
    This should be overloaded with a viable function
    """
    raise NotImplementedError

  def start_test(self, test_name, dqn_obj, depth_image_fcn, image_rate=1, heuristic=False):
    """
    Data for an entire test
    """

    best_sr, best_ep = dqn_obj.track.calc_best_performance()

    # create test data structure
    self.data = GraspTestData.TestData(
      [],                                       # trials
      test_name,                                # test_name
      dqn_obj.env.params.finger_width,          # finger width
      dqn_obj.env.params.finger_thickness,      # finger thickness
      heuristic,                                # is heuristic test
      dqn_obj.env.mj.set.bending_gauge.in_use,  # bending sensor
      dqn_obj.env.mj.set.palm_sensor.in_use,    # palm sensor
      dqn_obj.env.mj.set.wrist_sensor_Z.in_use, # wrist sensor
      dqn_obj.group_name,                       # run group name
      dqn_obj.run_name,                         # run name
      best_sr,                                  # best test success rate in sim
      best_ep,                                  # episode at best_sr
    )

    self.image_data = GraspTestData.TestData(
      [],                                       # trials
      test_name,                                # test_name
      dqn_obj.env.params.finger_width,          # finger width
      dqn_obj.env.params.finger_thickness,      # finger thickness
      heuristic,                                # is heuristic test
      dqn_obj.env.mj.set.bending_gauge.in_use,  # bending sensor
      dqn_obj.env.mj.set.palm_sensor.in_use,    # palm sensor
      dqn_obj.env.mj.set.wrist_sensor_Z.in_use, # wrist sensor
      dqn_obj.group_name,                       # run group name
      dqn_obj.run_name,                         # run name
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

  def finish_trial(self, request):
    """
    Finish a trial and save the results
    """

    # special cases where one flag implies another
    if request.s_h: request.t_h = 1 # ie True
    if request.t_h: request.lft = 1
    if request.drp: request.lft = 1

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
    self.current_trial.info = request.info
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
    self.current_trial_with_images.info = request.info
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

    if len(data.trials) == 0:
      print("WARNING: get_test_results() found 0 trials")
      return None

    # sort trial data
    for trial in data.trials:

      # ignore any object number over 100
      if trial.object_num > 100: continue

      num_trials += 1

      found = False
      for j in range(len(object_nums)):
        if object_nums[j] == trial.object_num:
          found = True
          break

      if not found:

        # create a new entry for this object
        new_entry = deepcopy(trial)
        new_entry.trial_num = 1
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
        entries[j].info += trial.info

    # create TestResults to return
    result = GraspTestData.TestResults()

    # now process trial data
    for i in range(len(entries)):

      if print_trials:
        print(f"Object num = {entries[i].object_num}, num trials = {entries[i].trial_num}, TH = {entries[i].target_height} SH = {entries[i].stable_height}")

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

    result.sphere_SR /= result.num_sphere
    result.cuboid_SR /= result.num_cuboid
    result.cylinder_SR /= result.num_cylinder
    result.cube_SR /= result.num_cube
    
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
      info_str += "\n"
    info_str += f"Sphere success rate: {results.sphere_SR:.4f}\n"
    info_str += f"cylinder success rate: {results.cylinder_SR:.4f}\n"
    info_str += f"cuboid success rate: {results.cuboid_SR:.4f}\n"
    info_str += f"cube success rate: {results.cube_SR:.4f}\n"
    info_str += "\n"
    info_str += f"Total number of trials: {results.num_trials}\n"
    info_str += f"Total number of objects: {results.num_objects}\n"
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
      
