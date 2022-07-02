#!/usr/bin/env python
# coding: latin-1
# I am Mohammad Omar, this module is builded to interface with the Driver MDD10A, to control two DC motors.
# the original code designed by Ingmar Stapel ,www.raspberry-pi-car.com to control two motors with a L298N H-Bridge
# The pins configuration for Model B Revision 1.0 

# How to Use this module: 1- creating an instance of the class. 2- call the Init function, 3- call commands functions
# Example:
# import MDD10A
# Motors = MDD10A.MDD10A()

# Import the libraries the class needs
import RPi.GPIO as io
io.setmode(io.BCM)

# Constant values, 
PWM_MAX                 = 100

# Disable warning from GPIO
io.setwarnings(False)

# Here we configure the GPIO settings for the left and right motors spinning direction.
# as described in the user manual of MDD10A http://www.robotshop.com/media/files/pdf/user-manual-mdd10a.pdf
# there are four input PWM1-DIR1-PWM2-DIR2
# WITH MAX Frequency 20 Hz, and it works as follow,
#		Input	DIR		Output-A	Output-B
#	PWM	 off	X		  off		  off
#	PWM	 on		off		  on		  off
#	PWM	 on		on		  off		  on

# The pins configuration for Model B Revision 1.0 
leftMotor_DIR_pin = 20
io.setup(leftMotor_DIR_pin, io.OUT)

rightMotor_DIR_pin = 26
io.setup(rightMotor_DIR_pin, io.OUT)

io.output(leftMotor_DIR_pin, False)

io.output(rightMotor_DIR_pin, False)


# Here we configure the GPIO settings for the left and right motors spinning speed. 

leftMotor_PWM_pin = 16
rightMotor_PWM_pin = 19

io.setup(leftMotor_PWM_pin, io.OUT)
io.setup(rightMotor_PWM_pin, io.OUT)

# MAX Frequency 20 Hz
leftMotorPWM = io.PWM(leftMotor_PWM_pin,20)
rightMotorPWM = io.PWM(rightMotor_PWM_pin,20)

leftMotorPWM.start(0)
leftMotorPWM.ChangeDutyCycle(0)

rightMotorPWM.start(0)
rightMotorPWM.ChangeDutyCycle(0)

leftMotorPower = 0
rightMotorPower = 0

def getMotorPowers():
	
	return (leftMotorPower,rightMotorPower)		

def setMotorLeft(power):

# SetMotorLeft(power)

# Sets the drive level for the left motor, from +1 (max) to -1 (min).

# This is a short explanation for a better understanding:
# SetMotorLeft(0)     -> left motor is stopped
# SetMotorLeft(0.75)  -> left motor moving forward at 75% power
# SetMotorLeft(-0.5)  -> left motor moving reverse at 50% power
# SetMotorLeft(1)     -> left motor moving forward at 100% power

	if power < 0:
		# Reverse mode for the left motor
		io.output(leftMotor_DIR_pin, False)
		pwm = -int(PWM_MAX * power)
		if pwm > PWM_MAX:
			pwm = PWM_MAX
	elif power > 0:
		# Forward mode for the left motor
		io.output(leftMotor_DIR_pin, True)
		pwm = int(PWM_MAX * power)
		if pwm > PWM_MAX:
			pwm = PWM_MAX
	else:
		# Stopp mode for the left motor
		io.output(leftMotor_DIR_pin, False)
		pwm = 0
#	print "SetMotorLeft", pwm
	leftMotorPower = pwm
	leftMotorPWM.ChangeDutyCycle(pwm)

def setMotorRight(power):

# SetMotorRight(power)

# Sets the drive level for the right motor, from +1 (max) to -1 (min).

# This is a short explanation for a better understanding:
# SetMotorRight(0)     -> right motor is stopped
# SetMotorRight(0.75)  -> right motor moving forward at 75% power
# SetMotorRight(-0.5)  -> right motor moving reverse at 50% power
# SetMotorRight(1)     -> right motor moving forward at 100% power

	if power < 0:
		# Reverse mode for the right motor
		io.output(rightMotor_DIR_pin, True)
		pwm = -int(PWM_MAX * power)
		if pwm > PWM_MAX:
			pwm = PWM_MAX
	elif power > 0:
		# Forward mode for the right motor
		io.output(rightMotor_DIR_pin, False)
		pwm = int(PWM_MAX * power)
		if pwm > PWM_MAX:
			pwm = PWM_MAX
	else:
		# Stopp mode for the right motor
		io.output(rightMotor_DIR_pin, False)
		pwm = 0
#	print "SetMotorRight", pwm
	rightMotorPower = pwm
	rightMotorPWM.ChangeDutyCycle(pwm)

def exit():
# Program will clean up all GPIO settings and terminates
	io.output(leftMotor_DIR_pin, False)
	io.output(rightMotor_DIR_pin, False)
	io.cleanup()