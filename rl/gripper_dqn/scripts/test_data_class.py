import numpy as np
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

class GraspTestData:

  # data structures for saving testing data
  step_data = namedtuple("step_data", ("step_num", "state_vector", "SI_state", "action"))
  image_data = namedtuple("image_data", ("step_num", "rgb", "depth"))

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
    num_trials: int
    num_objects: int
    avg_obj_num_trials: float
    success_rate: float
    avg_obj_success_rate: float
    sphere_SR: float
    cylinder_SR: float
    cuboid_SR: float
    cube_SR: float

  def __init__(self, test_name, dqn_obj, image_rate=1, heuristic=False):
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
      rgb, depth = get_depth_image()
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

    if data is None:
      data = self.data

    entries = []
    object_nums = []
    entry = ["obj_name", "object_num", "num_trials", "num_successes", "info_strings"]
    entry[0] = ""
    entry[1] = 0
    entry[2] = 0
    entry[3] = 0
    entry[4] = []

    if len(data.trials) == 0:
      print("get_test_results() found 0 trials")
      return None

    # sort trial data
    for trial in data.trials:

      found = False
      for j in range(len(object_nums)):
        if object_nums[j] == trial.object_num:
          found = True
          break

      if not found:

        # create a new entry
        new_entry = deepcopy(entry)
        new_entry[0] = trial.object_name
        new_entry[1] = trial.object_num
        new_entry[2] += 1
        new_entry[3] += trial.stable_height
        new_entry[4].append(trial.info)

        entries.append(new_entry)
        object_nums.append(trial.object_num)

      else:

        # add to the existing entry
        entries[j][2] += 1
        entries[j][3] += trial.stable_height
        entries[j].append(trial.info)

    # if True:
    #   for i in range(len(entries)):
    #     print(f"Object num = {entries[1]}, num trials = {entries[2]}")

    # now process trial data
    object_SRs = []
    object_trials = []
    total_successes = 0
    for i in range(len(entries)):

      total_successes += entries[i][3]
      this_SR = (entries[i][3] / float(entries[i][2]))
      object_SRs.append(this_SR)
      object_trials.append(entries[i][2])

      print(f"Object num = {entries[i][1]}, num trials = {entries[i][2]}, SR = {this_SR}")
  
    # round up
    total_SR = total_successes / float(len(data.trials))
    avg_obj_SR = np.mean(np.array(object_SRs))
    avg_obj_trials = np.mean(np.array(object_trials))

    return GraspTestData.TestResults(
      len(data.trials),        # num_trials
      len(object_nums),             # num_objects
      avg_obj_trials,               # avg_obj_num_trials
      total_SR,                     # success_rate
      avg_obj_SR,                   # avg_obj_success_rate
      0.0,                          # sphere_SR
      0.0,                          # cylinder_SR
      0.0,                          # cuboid_SR
      0.0,                          # cube_SR
    )

  def get_test_string(self, data=None):
    """
    Print out information about the current test
    """

    # by default use any current test data
    if data is None:
      data = self.data

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
    info_str += f"Loaded best SR: {data.best_SR}\n"

    results = self.get_test_results(data=data)

    if results is None: 
      info_str += "\nNO TRIAL DATA FOUND FOR THIS TEST"
      return info_str

    info_str += f"\nResults information:\n\n"
    info_str += f"Total number of trials: {results.num_trials}\n"
    info_str += f"Total number of objects: {results.num_objects}\n"
    info_str += f"Avg. trials per object: {results.avg_obj_num_trials:.4f}\n"
    info_str += f"Overall success rate: {results.success_rate:.4f}\n"
    info_str += f"Avg. success rate per object: {results.avg_obj_success_rate:.4f}\n"
    # info_str += f"Sphere success rate: {results.sphere_SR:.4f}\n"
    # info_str += f"cylinder success rate: {results.cylinder_SR:.4f}\n"
    # info_str += f"cuboid success rate: {results.cuboid_SR:.4f}\n"
    # info_str += f"cube success rate: {results.cube_SR:.4f}\n"

    return info_str