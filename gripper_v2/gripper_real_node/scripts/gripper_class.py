#!/home/luke/pyenv/py38_ros/bin/python

import serial
import struct
import time

def to_byte(x):
  # return bytes(x) # python 3
  return struct.pack("B", x)

def from_byte(x):
  (y,) = struct.unpack("<B", x)
  return y

class Gripper:
  """
  This class is used to control the 2nd prototype gripper
  """

  # define bytes for communication instructions
  motorCommandByte_m = to_byte(100)
  motorCommandByte_mm = to_byte(101)
  jointCommandByte_m_rad = to_byte(102)
  jointCommandByte_mm_deg = to_byte(103)
  stepCommandByte = to_byte(104)
  timedCommandByte_m = to_byte(105)

  # special communications
  homeByte = to_byte(110)
  powerSavingOnByte = to_byte(111)
  powerSavingOffByte = to_byte(112)
  stopByte = to_byte(113)
  resumeByte = to_byte(114)
  setSpeedByte = to_byte(115)
  debugOnByte = to_byte(116)
  debugOffByte = to_byte(117)
  printByte = to_byte(118)
  changeTimedActionByte = to_byte(119)
  changeTimedActionPubEarlyByte = to_byte(120)
  setGaugeHzByte = to_byte(121)
  setPublishHzByte = to_byte(122)
  setSerialHzByte = to_byte(123)
  setMotorHzByte = to_byte(124)
  homeBlockingByte = to_byte(125)

  # define bytes for communication protocol and error messages
  messageReceivedByte = to_byte(200)
  messageFailedByte = to_byte(201)
  targetNotReachedByte = to_byte(202)
  targetReachedByte = to_byte(203)
  invalidCommandByte = to_byte(204)
  debugMessageByte = to_byte(205)

  # signature bytes
  specialByte = to_byte(253)
  startMarkerByte = to_byte(254)
  endMarkerByte = to_byte(250)

  startEndSize = 3  # number of bytes for each of start/end markers

  debug = True  # are we in debug mode
  log_level = 1  # 0=disabled, 1=essential, 2=debug
  connected = False  # are we connected over Serial

  # ------------------------------------------------------------------------#

  class Command:
    """This sub-class contains commands to be sent to the gripper"""

    x = 0.0
    y = 0.0
    z = 0.0

    def __init__(self, units="m"):
      self.units = units
      self.check_units()

    def check_units(self):
      if self.units not in ["m", "mm", "m_rad", "mm_deg"]:
        raise RuntimeError(
          "Gripper command units must be either:\n"
          "\tm -> x,y,z are motor positions in metres\n"
          "\tmm -> x,y,z are motor positions in millimetres\n"
          "\tm_rad -> x,z are motor positions in metres, y is finger angle in radians\n"
          "\tmm_deg -> x,z are motor positions in millimetres, y is finger angle in degrees\n"
        )

    def listed(self):
      return [self.x, self.y, self.z]

  # ------------------------------------------------------------------------#

  class State:
    """Sub-class containing the last known state of the gripper"""

    information_byte = 0
    is_target_reached = True
    gauge1_data = 0.0
    gauge2_data = 0.0
    gauge3_data = 0.0
    gauge4_data = 0.0
    x_mm = -1.0
    y_mm = -1.0
    z_mm = -1.0

    def __init__(self):
      pass

  # ------------------------------------------------------------------------#

  class GripperException(Exception):
    pass

  # ------------------------------------------------------------------------#

  def __init__(self):
    self.command = self.Command(units="m")
    self.state = self.State()
    self.first_message_received = False

  def connect(self, com_port, baud_rate=115200):
    """
    Initiate a connection with the gripper over bluetooth
    """
    self.baud_rate = baud_rate
    tries = 0
    while tries < 5:
      try:
        self.serial = serial.Serial(com_port, self.baud_rate)
        self.com_port = com_port
        self.connected = True
        self.debug_print("Gripper is connected", log_level=1)
        return
      except serial.serialutil.SerialException as e:
        self.connected = False
        self.debug_print("Serial connection failed: " + repr(e), log_level=2)
        time.sleep(0.5)
        tries += 1
        self.debug_print(
          "Trying again...this is try number " + str(tries), log_level=2
        )
    self.debug_print(
      "Failed to get a connection with the gripper after "
      + str(tries)
      + " tries",
      log_level=1,
    )

  def debug_print(self, print_string, log_level=2):
    """
    This method prints an output string if we are in debug mode
    """

    if log_level <= self.log_level:
      print(print_string)

  def add_command_to_message(self, byte_msg):
    """
    Add the current values from the command to the byte message
    """

    # padding bytes (required for giga, messages must be multiples of 4)
    for i in range(3):
      byte_msg += to_byte(50)

    command_list = self.command.listed()
    for i in range(len(command_list)):
      byte_msg += bytearray(struct.pack("f", command_list[i]))

    return byte_msg

  def send_message(self, type="command", units=None):
    """
    This method publishes a method of the specified type
    """

    if not self.connected:
      self.debug_print(
        "Cannot send a message, gripper is not connected", log_level=2
      )
      return

    byte_msg = bytearray()

    # set the start markers
    for i in range(self.startEndSize):
      byte_msg += self.startMarkerByte

    if units is not None:
      self.command.units = units

    if type == "max":
      self.command.x = 52e-3
      self.command.y = 52e-3
      self.command.z = 165e-3
      type = "command"
    if type == "min":
      self.command.x = 133e-3
      self.command.y = 133e-3
      self.command.z = 5e-3
      type = "command"

    # fill in the main message
    if type == "command":

      self.command.check_units()

      if self.command.units == "m":
        byte_msg += bytearray(self.motorCommandByte_m)
      elif self.command.units == "mm":
        byte_msg += bytearray(self.motorCommandByte_mm)
      elif self.command.units == "m_rad":
        byte_msg += bytearray(self.jointCommandByte_m_rad)
      elif self.command.units == "mm_deg":
        byte_msg += bytearray(self.jointCommandByte_mm_deg)
      else:
        raise RuntimeError("invalid units for gripper command")
      
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "timed_command":

      self.command.check_units()

      if self.command.units == "m":
        byte_msg += bytearray(self.timedCommandByte_m)
      #   elif self.command.units == "mm":
      #       byte_msg += bytearray(self.motorCommandByte_mm)
      #   elif self.command.units == "m_rad":
      #       byte_msg += bytearray(self.jointCommandByte_m_rad)
      #   elif self.command.units == "mm_deg":
      #       byte_msg += bytearray(self.jointCommandByte_mm_deg)
      else:
        raise RuntimeError("invalid units for gripper timed command")
      
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "set_speed":

      byte_msg += bytearray(self.setSpeedByte)
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "home":
      byte_msg += bytearray(self.homeByte)

    elif type == "home_blocking":
      byte_msg += bytearray(self.homeBlockingByte)

    elif type == "power_saving_off":
      byte_msg += bytearray(self.powerSavingOffByte)

    elif type == "power_saving_on":
      byte_msg += bytearray(self.powerSavingOnByte)

    elif type == "stop":
      byte_msg += bytearray(self.stopByte)

    elif type == "resume":
      byte_msg += bytearray(self.resumeByte)

    elif type == "debug_on":
      byte_msg += bytearray(self.debugOnByte)

    elif type == "debug_off":
      byte_msg += bytearray(self.debugOffByte)

    elif type == "print":
      byte_msg += bytearray(self.printByte)

    elif type == "change_timed_action":
      byte_msg += bytearray(self.changeTimedActionByte)
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "change_timed_action_early_pub":
      byte_msg += bytearray(self.changeTimedActionPubEarlyByte)
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "set_gauge_hz":
      byte_msg += bytearray(self.setGaugeHzByte)
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "set_publish_hz":
      byte_msg += bytearray(self.setPublishHzByte)
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "set_serial_hz":
      byte_msg += bytearray(self.setSerialHzByte)
      byte_msg = self.add_command_to_message(byte_msg)

    elif type == "set_motor_hz":
      byte_msg += bytearray(self.setMotorHzByte)
      byte_msg = self.add_command_to_message(byte_msg)

    else:
      self.debug_print(
        "incorrect type given to gripper_class send_message()", log_level=1
      )
      return

    # set the end markers
    for i in range(self.startEndSize):
      byte_msg += self.endMarkerByte

    # send the message
    self.serial.write(byte_msg)

  def get_serial_message(self, timeout=0.5):
    """
    This method gets the next message in the serial buffer, and will wait
    until the timeout for it to arrive
    """

    try:

      if not self.connected:
        self.debug_print(
          "Gripper is not connected, get_serial_message(...) aborting",
          log_level=2,
        )
        return "gripper not connected"

      # are there no bytes in the buffer
      if self.serial.in_waiting == 0:
        return "empty buffer"

      t0 = time.time()
      t1 = time.time()

      message_started = False
      data_received = 0
      signature_count = 0

      received_bytes = bytearray()

      i = 0

      # test = bytearray()

      while timeout > t1 - t0:

        t1 = time.time()

        if self.serial.in_waiting > 0:
          x = self.serial.read()

          # test += x

          # are we in the middle of reading the message
          if message_started:
            received_bytes += x
            data_received += 1

          # the below logic checks for start and end signatures
          if not message_started:
            if x == self.startMarkerByte:
              signature_count += 1
              if signature_count == self.startEndSize:
                message_started = True
                signature_count = 0
            else:
              signature_count = 0
          # else the message has started
          else:
            if x == self.endMarkerByte:
              signature_count += 1
              if signature_count == self.startEndSize:
                data_received -= self.startEndSize
                # remove the end marker from the message
                for i in range(self.startEndSize):
                  received_bytes.pop()
                # print("Message received")
                # print(list(test))
                self.first_message_received = True
                return received_bytes
            else:
              signature_count = 0

        #   else:
        #       return "ran out of buffer"

      return "timeout"

    except IOError as e:

      print("gripper.get_serial_message error:", e)
      self.connect(self.com_port)

      # recursive loop
      return self.get_serial_message(timeout=timeout)

  def update_state(self):
    """
    This method updates the state of the gripper
    """

    faulty_messages = 0
    max_faults = 10

    debug = False

    # loop until buffer is empty
    while True:

      output = self.get_serial_message()

      # if buffer is empty or something went wrong
      if type(output) == str:
        if output != "empty buffer":
          self.debug_print("Error in gauge read: " + output, log_level=2)
        break

      # message length for a normal message
      desiredByteLength = 32

      informationByte = output[0]
      isTargetReached = output[1]

      # print("information byte is", informationByte)
      # print("isTargetReached is", isTargetReached)
      # print("Remaining bytes are", list(output[2:]), "total message length is", len(output))

      # check if we have received a debug message
      if informationByte == from_byte(self.debugMessageByte):
        try:
          debug_str = output[1:].decode("utf-8")
          # print the message to the regular terminal, then continue
          print("Gripper debug message is:")
          strs = debug_str.split("\n")
          for s in strs:
            print(s)
        except Exception as e:
          print(e)
        continue

      # check that output fits our requirements (getting errors of 3,8)
      if len(output) != desiredByteLength:
        if debug:
          error_str = (
            "Wrong size! The length was %d when it should have been %d"
            % (len(output), desiredByteLength)
          )
          self.debug_print(error_str, log_level=2)
        faulty_messages += 1
        if faulty_messages > max_faults:
          # clear the buffer
          self.serial.read_all()
          self.debug_print(
            "Clearing the buffer, flushing away all pending messages",
            log_level=1,
          )
        continue

      # python 2.7, struct returns a tuple eg (101,), we want int
      # '<l' is "<" little endian, "l" long
      # '<f' is "<" little endian, "f" float
      s = 4
      (reading1,) = struct.unpack("<l", output[s : s + 4])
      s += 4
      (reading2,) = struct.unpack("<l", output[s : s + 4])
      s += 4
      (reading3,) = struct.unpack("<l", output[s : s + 4])
      s += 4
      (reading4,) = struct.unpack("<l", output[s : s + 4])
      s += 4
      (xPosition,) = struct.unpack("<f", output[s : s + 4])
      s += 4
      (yPosition,) = struct.unpack("<f", output[s : s + 4])
      s += 4
      (zPosition,) = struct.unpack("<f", output[s : s + 4])
      s += 4

      # save state
      self.state.information_byte = informationByte
      self.state.is_target_reached = isTargetReached
      self.state.gauge1_data = reading1
      self.state.gauge2_data = reading2
      self.state.gauge3_data = reading3
      self.state.gauge4_data = reading4
      self.state.x_mm = xPosition
      self.state.y_mm = yPosition
      self.state.z_mm = zPosition

      self.connected = True

    return self.state
