#!/usr/bin/env python

import serial
import struct
import time

def to_byte(x):
    return struct.pack("B", x)

class Gripper:
    '''This class is used to control the 2nd prototype gripper'''

    # define bytes for communication instructions
    # sendCommandByte = bytes([100]) python 3+
    sendCommandByte = to_byte(100)
    homeByte = to_byte(101)
    powerSavingOnByte = to_byte(102)
    powerSavingOffByte = to_byte(103)
    stopByte = to_byte(104)
    resumeByte = to_byte(105)
    
    # define bytes for communication protocol
    messageReceivedByte = to_byte(200)
    messageFailedByte = to_byte(201)
    targetNotReachedByte = to_byte(202)
    targetReachedByte = to_byte(203)
    
    specialByte = to_byte(253)
    startMarkerByte = to_byte(254)
    endMarkerByte = to_byte(255)
    
    startEndSize = 2          # number of bytes for each of start/end markers
    
    debug = True                    # are we in debug mode
    connected = False
    gauge_data_array = []
    gauge1_data = []
    gauge2_data = []
    gauge3_data = []
    
    #------------------------------------------------------------------------#
    class Command:
        '''This sub-class contains commands to be sent to the gripper'''
        radius = 0.0        # in mm (from 50mm to 135mm)
        angle = 0.0         # in deg (from -40deg to +40deg)
        palm = 0.0          # in mm (from 0 to 160mm)
        def __init__(self):
            self.filler = "filled"
        def listed(self):
            return [self.radius, self.angle, self.palm]
    #------------------------------------------------------------------------#
    class State:
      """Sub-class containing the last known state of the gripper"""
      is_target_reached = True
      gauge1_data = 0.0
      gauge2_data = 0.0
      gauge3_data = 0.0
      x_mm = -1.0
      y_mm = -1.0
      z_mm = -1.0
      def __init__(self):
        pass
    #------------------------------------------------------------------------#
    class GripperException(Exception):
      pass
    #------------------------------------------------------------------------#
            
    def __init__(self):
      self.command = self.Command()
      self.state = self.State()

    def connect(self, com_port):
        self.serial = serial.Serial(com_port, 115200)
        
    def debug_print(self,print_string):
        '''This method prints an output string if we are in debug mode'''
        if self.debug == True:
            print(print_string)
        return

    def get_state(self):
      return self.state

    def send_message(self, type="command"):
        """This method publishes a method of the specified type"""

        byte_msg = bytearray()

        # set the start markers
        for i in range(self.startEndSize):
            byte_msg += self.startMarkerByte

        # fill in the main message
        if type == "command":
            byte_msg += bytearray(self.instructionByte)
            command_list = self.command.listed()
            for i in range(len(command_list)):
                byte_msg += bytearray(struct.pack("f", command_list[i]))

        elif type == "home":
            byte_msg += bytearray(self.homeByte)

        elif type == "power_saving_off":
            byte_msg += bytearray(self.powerSavingOffByte)

        elif type == "power_saving_on":
            byte_msg += bytearray(self.powerSavingOnByte)

        elif type == "stop":
            byte_msg += bytearray(self.stopByte)

        elif type == "resume":
            byte_msg += bytearray(self.resumeByte)

        else:
            print("incorrect type given to send_message")
            return

        # set the end markers
        for i in range(self.startEndSize):
            byte_msg += self.endMarkerByte

        # send the message
        self.serial.write(byte_msg)
            
    def send_command(self, instructionByte=0):
        '''This method publishes the command to the gripper via the serial
        port. Returns true if the command is successfully received'''
        
        # default input
        if instructionByte == 0:
            instructionByte = self.sendCommandByte
        
        # get the command in list form
        command_list = self.command.listed()
        
        byte_msg = bytearray()

        # set the start markers
        for i in range(self.startEndSize):
            byte_msg += self.startMarkerByte
        
        # next the instruction byte
        byte_msg += bytearray(instructionByte)

        # then the data
        for i in range(len(command_list)):
            byte_msg += bytearray(struct.pack("f", command_list[i]))
            
        # finally the end marker
        for i in range(self.startEndSize):
            byte_msg += self.endMarkerByte
        
        # now flush the input buffer
        # self.serial.flushInput()
        self.debug_print("Sending command")
        
        # send the message
        self.serial.write(byte_msg)

        # TEMPORARY MEASURE
        return True
        
        # wait for a response
        response = self.get_next_output()
        
        if response == self.messageReceivedByte:
            self.debug_print("Message received successfully")
            return True
        else:
            self.debug_print("Message sending failed")
            if response == self.messageFailedByte:
                self.debug_print("Received failure byte from Arduino")
            elif response == "timeout":
                self.debug_print("No message received before timeout")
            else:
                self.debug_print("Unknown error, response was: " + str(response))
            return False
        
    def get_serial_message(self, timeout=3.0):
        '''This method gets the next message in the serial buffer, and will wait
        until the timeout for it to arrive'''
        
        # are there no bytes in the buffer
        if self.serial.in_waiting == 0:
            return "empty buffer"
        
        t0 = time.clock()
        t1 = time.clock()
        
        message_started = False
        data_received = 0
        signature_count = 0
        
        received_bytes = bytearray()

        i = 0

        while timeout > t1 - t0:

            t1 = time.clock()

            if self.serial.in_waiting > 0:
                x = self.serial.read()

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
                            return received_bytes
                    else:
                        signature_count = 0
                    
        return "timeout"

    def update_state(self):
        """This method updates the state of the gripper"""

        # loop until buffer is empty
        while True:

            output = self.get_serial_message()
            
            # if buffer is empty or something went wrong
            if type(output) == str:
                if output != "empty buffer":
                    self.debug_print("Error in gauge read: ")
                    self.debug_print(output)
                return

            # check that output fits our requirements (getting errors of 3,8)
            if len(output) != 25:
                print("Wrong size! The length was", len(output), 
                    "when it should have been 25")
                continue

            isTargetReached = output[0]
            # python 2.7, struct returns a tuple eg (101,), we want int
            # '<l' is "<" little endian, "l" long
            # '<f' is "<" little endian, "f" float
            (reading1,) = struct.unpack("<l", output[1:5])
            (reading2,) = struct.unpack("<l", output[5:9])
            (reading3,) = struct.unpack("<l", output[9:13])
            (xPosition,) = struct.unpack("<f", output[13:17])
            (yPosition,) = struct.unpack("<f", output[17:21])
            (zPosition,) = struct.unpack("<f", output[21:25])

            # save state
            self.state.is_target_reached = isTargetReached
            self.state.gauge1_data = reading1
            self.state.gauge2_data = reading2
            self.state.gauge3_data = reading3
            self.state.x_mm = xPosition
            self.state.y_mm = yPosition
            self.state.z_mm = zPosition

            self.connected = True

            # print("The gauge readings are", reading1,";",reading2,";",reading3)
    
    def read_gauge(self):
        '''This method reads a strain gauge message from the serial input'''
        
        output = self.get_serial_message()
        
        # if something went wrong
        if type(output) == str:
            if output != "empty buffer":
                self.debug_print("Error in gauge read: ")
                self.debug_print(output)
                return "error", False, False, False
            else:
                return "empty", False, False, False

        isTargetReached = output[0]
        # reading1 = int.from_bytes(output[1:5], byteorder="little", signed=True)
        # reading2 = int.from_bytes(output[5:9], byteorder="little", signed=True)
        # reading3 = int.from_bytes(output[9:], byteorder="little", signed=True)

        # python 2.7, struct returns a tuple eg (101,), we want int
        # '<l' is "<" little endian, "l" long

        if len(output) != 13:
            print("Too short! The length was", len(output))
            return "too short", False, False, False

        (reading1,) = struct.unpack("<l", output[1:5])
        (reading2,) = struct.unpack("<l", output[5:9])
        (reading3,) = struct.unpack("<l", output[9:13])

        # print("The gauge readings are", reading1,";",reading2,";",reading3)
        
        return isTargetReached, reading1, reading2, reading3

    def update_gauge_data(self):
        # read until the buffer is empty
        while True:
            isTargetReached, reading1, reading2, reading3 = self.read_gauge()
            # if we get three False then the buffer is empty (or an error)
            if not reading1 and not reading2 and not reading3:
                return
            # add readings to our plot and loop, avoid erroneous zeros
            if reading1 != 0: self.gauge1_data.append(reading1)
            if reading2 != 0: self.gauge2_data.append(reading2)
            if reading3 != 0: self.gauge3_data.append(reading3)

    def live_gauge_output(self):
        """This method live updates a graph with the gauge output from a given
        gauge"""
                
        def animate(i):
            
            # update_data(self.gauge_data_array)
            # request_data(self.gauge_data_array)
            self.update_gauge_data()

            global stop_thread

            # number of data points shown on graph
            if stop_thread:
                D = 0
                stopping = True
            else:
                D = 50
                stopping = False
            n1 = len(self.gauge1_data)
            n2 = len(self.gauge2_data)
            n3 = len(self.gauge3_data)

            y1 = self.gauge1_data[-D:]
            y2 = self.gauge2_data[-D:]
            y3 = self.gauge3_data[-D:]

            m1 = len(y1)
            m2 = len(y2)
            m3 = len(y3)

            x1 = list(range(n1-m1, n1))
            x2 = list(range(n2-m2, n2))
            x3 = list(range(n3-m3, n3))

            ax1.clear()
            ax2.clear()
            ax3.clear()

            ax1.plot(x1, y1, 'c', label="Gauge 1")
            ax2.plot(x2, y2, 'm', label="Gauge 2")
            ax3.plot(x3, y3, 'y', label="Gauge 3")

            # ax1.legend()
            # ax2.legend()
            # ax3.legend()
            ax1.set_title("Gauge 1")
            ax2.set_title("Gauge 2")
            ax3.set_title("Gauge 3")

            plt.autoscale()

            if stopping:
                print("Stopping thread")
                strFile = "gauge_plot.png"
                if os.path.isfile(strFile):
                    os.remove(strFile)  # Opt.: os.system("rm "+strFile)
                plt.savefig(strFile)
                plt.close()
                print("Figure saved")
                filename = "test2"
                # np_data = np.array([mygripper.gauge1_data,
                #                                mygripper.gauge2_data,
                #                                mygripper.gauge3_data], dtype="object")
                np.savetxt("gauge1" + ".dat", np.array(mygripper.gauge1_data))
                np.savetxt("gauge2" + ".dat", np.array(mygripper.gauge1_data))
                np.savetxt("gauge3" + ".dat", np.array(mygripper.gauge1_data))
                # self.plot_final_gauge()
                exc = ctypes.py_object(SystemExit)
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(plot_thread.ident), exc)
                if res == 0:
                    raise ValueError("nonexistent thread id")
                elif res > 1:
                    # """if it returns a number greater than one, you're in trouble,
                    # and you should call it again with exc=NULL to revert the effect"""
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(plot_thread.ident, None)
                    raise SystemError("PyThreadState_SetAsyncExc failed")

        # style.use('fivethirtyeight')

        self.serial.flush()

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        fig.set_size_inches(6,7)
        fig.tight_layout(pad=3.0, h_pad=1.5)
        # fig.set_title("Strain gauge readings")
        # fig.ylabel("24bit reading")
        # fig.xlable("t")
        
        ani = animation.FuncAnimation(fig, animate, interval=100)
        plt.show()

    def plot_final_gauge(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.set_size_inches(6, 7)
        fig.tight_layout(pad=2.0, h_pad=1.5)

        # number of data points shown on graph
        n1 = len(self.gauge1_data)
        n2 = len(self.gauge2_data)
        n3 = len(self.gauge3_data)

        y1 = self.gauge1_data[:]
        y2 = self.gauge2_data[:]
        y3 = self.gauge3_data[:]

        m1 = len(y1)
        m2 = len(y2)
        m3 = len(y3)

        x1 = list(range(n1 - m1, n1))
        x2 = list(range(n2 - m2, n2))
        x3 = list(range(n3 - m3, n3))

        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.plot(x1, y1, 'c', label="Gauge 1")
        ax2.plot(x2, y2, 'm', label="Gauge 2")
        ax3.plot(x3, y3, 'y', label="Gauge 3")

        # ax1.legend()
        # ax2.legend()
        # ax3.legend()
        ax1.set_title("Gauge 1")
        ax2.set_title("Gauge 2")
        ax3.set_title("Gauge 3")

        plt.autoscale()
        plt.savefig("gauge_plot.png")

        return

    def live_output_2(self):

        self.serial.flush()
        self.update_gauge_data()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        X = list(range(len(self.gauge_data_array)))
        Y = self.gauge_data_array

        # # X = np.linspace(0,2,1000)
        # # Y = X**2 + np.random.random(X.shape)

        graph, = ax.plot(X, Y)

        fig.canvas.draw()
        plt.show()

        # graph, = plt.plot(X, Y)
        # plt.show()

        while True:

            self.update_gauge_data()
            X = list(range(len(self.gauge_data_array)))
            Y = self.gauge_data_array

            if len(X) > 50:
                X = X[-50:]
                Y = Y[-50:]

            # Y = X**2 + np.random.random(X.shape)

            graph.set_xdata(X)
            graph.set_ydata(Y)

            ax.relim()
            ax.autoscale_view(True, True, True)

            # fig.canvas.draw()
            # time.sleep(0.01)
            # # plt.pause(0.01)
            # plt.show()

            # plt.plot(X,Y)

            plt.draw()
            plt.pause(0.01)
            plt.show()

        # # fig = plt.figure()
        # # ax1 = fig.add_subplot(1,1,1)

        # # ani = animation.FuncAnimation(fig, animate, interval=100)
        # # plt.show()