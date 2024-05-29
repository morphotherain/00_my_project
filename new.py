import RPi.GPIO as GPIO
import time
import smbus
import cv2
import numpy as np
from threading import Thread
import pyautogui
import subprocess

# GPIO引脚定义
BUZZER_PIN = 11
GREEN_LED_PIN = 12
RED_LED_PIN = 13
JOYSTICK_X_PIN = 0
JOYSTICK_Y_PIN = 1
JOYSTICK_BTN_PIN = 2
BUTTON_LEFT_PIN = 14
BUTTON_RIGHT_PIN = 15
TRIG_PIN = 16
ECHO_PIN = 18

# MPU6050定义
MPU6050_ADDR = 0x68
POWER_MGMT_1 = 0x6b
POWER_MGMT_2 = 0x6c

# 摄像头定义
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = 'haarcascade_eye.xml'

# 初始化SMBus
bus = smbus.SMBus(1)

# 初始化GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 设置GPIO引脚
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
GPIO.setup(RED_LED_PIN, GPIO.OUT)
GPIO.setup(BUTTON_LEFT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_RIGHT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

# MPU6050初始化
def init_mpu6050():
    bus.write_byte_data(MPU6050_ADDR, POWER_MGMT_1, 0)

def read_mpu6050():
    gyro_xout = read_word_2c(0x43)
    gyro_yout = read_word_2c(0x45)
    gyro_zout = read_word_2c(0x47)
    accel_xout = read_word_2c(0x3b)
    accel_yout = read_word_2c(0x3d)
    accel_zout = read_word_2c(0x3f)
    return (gyro_xout, gyro_yout, gyro_zout, accel_xout, accel_yout, accel_zout)

def read_word_2c(adr):
    val = (bus.read_byte_data(MPU6050_ADDR, adr) << 8) + bus.read_byte_data(MPU6050_ADDR, adr + 1)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

# 摇杆初始化
def init_joystick():
    bus.write_byte_data(0x48, 0x40, 0x03)

def read_joystick():
    x = bus.read_byte_data(0x48, JOYSTICK_X_PIN)
    y = bus.read_byte_data(0x48, JOYSTICK_Y_PIN)
    btn = GPIO.input(JOYSTICK_BTN_PIN)
    return (x, y, btn)

# 蜂鸣器函数
def buzzer_on():
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def buzzer_off():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)

# LED函数
def led_green_on():
    GPIO.output(GREEN_LED_PIN, GPIO.HIGH)

def led_green_off():
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

def led_red_on():
    GPIO.output(RED_LED_PIN, GPIO.HIGH)

def led_red_off():
    GPIO.output(RED_LED_PIN, GPIO.LOW)

def led_red_blink():
    while True:
        GPIO.output(RED_LED_PIN, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(RED_LED_PIN, GPIO.LOW)
        time.sleep(0.5)

# 按键函数
def read_button_left():
    return GPIO.input(BUTTON_LEFT_PIN) == GPIO.LOW

def read_button_right():
    return GPIO.input(BUTTON_RIGHT_PIN) == GPIO.LOW

# 摄像头函数
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
cap = cv2.VideoCapture(0)

def fatigue_detection():
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) == 0:
                print("检测到疲劳!")
                led_red_blink()
        time.sleep(1)

# 距离检测函数
def distance_measurement():
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    start_time = time.time()
    stop_time = time.time()

    while GPIO.input(ECHO_PIN) == 0:
        start_time = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        stop_time = time.time()

    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2
    return distance

# 游戏控制函数
def game_control():
    while True:
        x, y, btn = read_joystick()
        # 摇杆控制WASD和空格
        if x < 100:
            pyautogui.keyDown('a')
        elif x > 150:
            pyautogui.keyDown('d')
        else:
            pyautogui.keyUp('a')
            pyautogui.keyUp('d')
        if y < 100:
            pyautogui.keyDown('w')
        elif y > 150:
            pyautogui.keyDown('s')
        else:
            pyautogui.keyUp('w')
            pyautogui.keyUp('s')
        if btn == 0:
            pyautogui.press
            pyautogui.press('space')

        # 按键控制鼠标点击
        if read_button_left():
            pyautogui.click(button='left')
        if read_button_right():
            pyautogui.click(button='right')
        
        time.sleep(0.1)

def main():
    init_mpu6050()
    init_joystick()
    led_green_on()

    # 启动游戏
    game_path = '/path/to/your/game/executable'  # 替换为实际的游戏路径
    subprocess.Popen(game_path, shell=True)

    # 启动疲劳检测线程
    fatigue_thread = Thread(target=fatigue_detection)
    fatigue_thread.start()

    # 启动游戏控制线程
    game_control_thread = Thread(target=game_control)
    game_control_thread.start()

    try:
        while True:
            # 读取并打印MPU6050数据
            gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z = read_mpu6050()
            print(f"Gyro: {gyro_x}, {gyro_y}, {gyro_z}")
            print(f"Accel: {accel_x}, {accel_y}, {accel_z}")

            # 距离检测并报警
            distance = distance_measurement()
            print(f"Distance: {distance} cm")
            if distance < 50:  # 距离阈值可以根据需要调整
                print("距离过近！")
                buzzer_on()
                led_red_on()
            else:
                buzzer_off()
                led_red_off()

            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
