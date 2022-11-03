/* Software License Agreement (BSD License)
*
* Copyright (c) 2014, Robotiq, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above
* copyright notice, this list of conditions and the following
* disclaimer in the documentation and/or other materials provided
* with the distribution.
* * Neither the name of Robotiq, Inc. nor the names of its
* contributors may be used to endorse or promote products derived
* from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
* Copyright (c) 2014, Robotiq, Inc
*/

/*
 *  \file main.c
 *  \date June 18, 2014
 *  \author Jonathan Savoie <jonathan.savoie@robotic.com>
 *  \maintener Nicolas Lauzier <nicolas@robotiq.com>
 */

#include <string.h>
#include <stdio.h>
#include <time.h>

#include "rq_sensor_com.h"
// #include "rq_sensor_state.h" // don't include this file! Conflicts with function declarations below in extern "C"
#include "rq_int.h"
#include "Thread/rq_thread.h"

#include <ros/ros.h>
#include <geometry_msgs/Wrench.h>

/* here we copy rq_sensor_state.h, but we declare functions as C style for C++ compiler */
enum rq_sensor_state_values 
{
	RQ_STATE_INIT,         ///< State that initialize the com. with the sensor
	RQ_STATE_READ_INFO,    ///< State that reads the firmware version,
	                       ///< serial number and production year
	RQ_STATE_START_STREAM, ///< State that start the sensor in streaming
	                       ///< mode
	RQ_STATE_RUN           ///< State that reads the streaming data from 
		                   ///< the sensor
};

// declare functions we need from driver C library
extern "C" {

  INT_8 rq_sensor_state(void);
  void rq_state_get_command(INT_8 const * const name, INT_8 * const  value);
  void rq_set_zero(void);
  enum rq_sensor_state_values rq_sensor_get_current_state(void);
  bool rq_state_got_new_message(void);
  float rq_state_get_received_data(UINT_8 i);

};
/* end copy of rq_sensor_state.h */

/**
 * \fn static void decode_message_and_do(char *buff)
 * \brief Decode the message received and execute the operation.
 * Accepted string 	: GET PYE
 * 					: GET FMW
 * 					: GET SNU
 * 					: SET ZRO
 * No other message will be accepted
 * \param *buff, String to decode and execute
 */
/*static void decode_message_and_do(UINT_8 *buff){
	if(buff == NULL || strlen(buff) < 7){
		return;
	}
	UINT_8 get_or_set[3];
	strncpy(get_or_set, &buff[0], 3);
	if(strstr(get_or_set, "GET")){
		UINT_8 nom_get_var[4];
		strncpy(nom_get_var, &buff[4], strlen(buff) -3);
		UINT_8 buffer[512];
		rq_state_get_command(nom_get_var, buffer);
		//
		//  Here comes the code to resend the high level data.
		//
	}
	else if (strstr(buff, "SET ZRO")){
		rq_set_zero();
	}
	//
	// Here comes the code to empty the string(buff).
	//
}*/

/**
 * \fn static void wait_for_other_connection()
 * \brief Function who wait for another connection
 */
static void wait_for_other_connection(){
	INT_8 ret;
	struct timespec tim;
	tim.tv_sec = 1;
	tim.tv_nsec = 0L;

	while(1){
		nanosleep(&tim, (struct timespec *)NULL);
		ret = rq_sensor_state();
		if(ret == 0){
			break;
		}
	}
}

/**
 * \fn void get_data(void)
 * \brief Function to retrieve the power applied to the sensor
 * \param chr_return String to return forces applied
 */
static void get_data(INT_8 * chr_return){
	INT_8 i;
	INT_8 floatData[50];
	for(i = 0; i < 6; i++){
		sprintf(floatData, "%f", rq_state_get_received_data(i));
		if(i == 0){
			strcpy(chr_return, "( ");
			strcat(chr_return, floatData);
		}
		else{
			strcat(chr_return," , ");
			strcat(chr_return,floatData);
		}
		if(i == 5){
			strcat(chr_return, " )");
		}
	}
}

void fill_wrench_message(geometry_msgs::Wrench& msg)
{
  /* get force torque information from the sensor and fill a ROS message */

  msg.force.x = rq_state_get_received_data(0);
  msg.force.y = rq_state_get_received_data(1);
  msg.force.z = rq_state_get_received_data(2);
  msg.torque.x = rq_state_get_received_data(3);
  msg.torque.y = rq_state_get_received_data(4);
  msg.torque.z = rq_state_get_received_data(5);
}

int main(int argc, char **argv) {

	 //IF can't connect with the sensor wait for another connection
	INT_8 ret = rq_sensor_state();
	if(ret == -1){
		wait_for_other_connection();
	}

	//Read high-level informations
	ret = rq_sensor_state();
	if(ret == -1){
		wait_for_other_connection();
	}

	//Initialize connection with the client
	ret = rq_sensor_state();
	if(ret == -1){
		wait_for_other_connection();
	}

	/*
	 * Here comes the code to establish a connection with the application
	*/

  // create ros node and publisher to send out sensor data
  ros::init(argc, argv, "ft_sensor_node");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<geometry_msgs::Wrench>("/gripper/real/ftsensor", 10);
  geometry_msgs::Wrench wrench_msg;
  ros::Rate rate(20); // 10 Hz data stream

	//INT_8 buffer[512]; //Init of the variable receiving the message
	INT_8 bufStream[512];


 	// while(1){
  while (ros::ok()) {

 		/*strcpy(buffer,"");
 		*  // Here we receive the message of the application to read
 		*  // high level variable.
 		*if(strcmp(buffer, "") != 0)
 		*{
 		*	decode_message_and_do(buffer);
 		*}
 		*/

 		ret = rq_sensor_state();
 		if(ret == -1){
      ROS_INFO("Waiting for other connection");
 			wait_for_other_connection();
 		}
 		if(rq_sensor_get_current_state() == RQ_STATE_RUN){
 			strcpy(bufStream,"");
 			get_data(bufStream);
 			// printf("%s\n", bufStream); // don't want to print to terminal
 			/*
 			 * Here comes the code to send the data to the application.
 			*/

      fill_wrench_message(wrench_msg);
      pub.publish(wrench_msg);

      // for testing
      ROS_INFO_STREAM("ftsensor z = " << wrench_msg.force.z);

      rate.sleep();
 		}
 	}

  ROS_INFO("ftsensor node shutting down now");
  
 	return 0;
 }
