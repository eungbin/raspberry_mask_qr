import gpio as GPIO
import time


def servoMotor(pin, degree, t):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)

    pwm.start(3)
    time.sleep(t)

    while(True):
        pwm.ChangeDutyCycle(degree)
        time.sleep(t)
        pwm.stop()
        GPIO.cleanup(pin)

servoMotor(16, 8, 1)