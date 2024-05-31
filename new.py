import RPi.GPIO as GPIO
import time
import smbus
import math
import cv2
import numpy as np
from threading import Thread
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController, Key



mouse = MouseController()
keyboard = KeyboardController()


# GPIO引脚定义
BUZZER_PIN = 12
GREEN_LED_PIN = 23
RED_LED_PIN = 26
BUTTON_LEFT_PIN = 22
BUTTON_RIGHT_PIN = 11
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

def dist(a, b):
    return math.sqrt((a * a) + (b * b))

def get_y_rotation(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)

def get_x_rotation(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)


def read_mpu6050():
    accel_xout = read_word_2c(0x3b)
    accel_yout = read_word_2c(0x3d)
    accel_zout = read_word_2c(0x3f)
    accel_xout_scaled = accel_xout / 16384.0
    accel_yout_scaled = accel_yout / 16384.0
    accel_zout_scaled = accel_zout / 16384.0

    x = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)
    y = get_y_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)
    
    return (x,y)

def read_word_2c(adr):
    val = (bus.read_byte_data(MPU6050_ADDR, adr) << 8) + bus.read_byte_data(MPU6050_ADDR, adr + 1)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

# 摇杆初始化
def init_joystick():
    bus.write_byte_data(0x48, 0x40, 0x03)

address = 0x48

def read(chn):
    try:
        if chn == 0:
            bus.write_byte(address,0x40)
        if chn == 1:
            bus.write_byte(address,0x41)
        if chn == 2:
            bus.write_byte(address,0x42)
        if chn == 3:
            bus.write_byte(address,0x43)
        bus.read_byte(address)
    except Exception as e:
        print ("Address: %s" % address)
        print (e)
    return bus.read_byte(address)

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
#face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
#eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
#cap = cv2.VideoCapture(0)

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

# 初始化pynput控制器
mouse = MouseController()
keyboard = KeyboardController()

def read_joystick_and_control():
    state = ['home', 'up', 'down', 'left', 'right', 'pressed']
    i = 0
    if read(1) <= 30:
        i = 1
        keyboard.press('w')
        keyboard.release('s')
    elif read(1) >= 225:
        i = 2
        keyboard.press('s')
        keyboard.release('w')
    elif read(0) >= 225:
        i = 3
        keyboard.press('a')
        keyboard.release('d')
    elif read(0) <= 30:
        i = 4
        keyboard.press('d')
        keyboard.release('a')
    elif read(2) == 0 and read(1) == 128 and read(0) > 100 and read(0) < 130:
        i = 5
        keyboard.press(Key.space)
        keyboard.release(Key.space)
    elif read(0) - 125 < 15 and read(0) - 125 > -15 and read(1) - 125 < 15 and read(1) - 125 > -15 and read(2) == 255:
        i = 0
        keyboard.release('w')
        keyboard.release('s')
        keyboard.release('a')
        keyboard.release('d')

    # 按键控制鼠标点击
    if read_button_left():
        print(f"leftclick")
        mouse.click(Button.left, 1)
    if read_button_right():
        print(f"rightclick")
        mouse.click(Button.right, 1)
    
    return state[i]

def game_control():
    while True:
        state = read_joystick_and_control()
        ## print(f"Joystick state: {state}")
        time.sleep(0.1)


def main():
    init_mpu6050()
    init_joystick()
    # led_green_on()

    # 启动游戏
    # game_path = '/path/to/your/game/executable'  # 替换为实际的游戏路径
    # subprocess.Popen(game_path, shell=True)

    # 启动疲劳检测线程
    # fatigue_thread = Thread(target=fatigue_detection)
    # fatigue_thread.start()

    # 启动游戏控制线程
    game_control_thread = Thread(target=game_control)
    game_control_thread.start()

    try:
        while True:
            # 读取并打印MPU6050数据
            r_x, r_y = read_mpu6050()
            print(f"x, y: {r_x}, {r_y}")

            # 距离检测并报警
            ##distance = distance_measurement()
            ##print(f"Distance: {distance} cm")
            ## if distance < 50:  # 距离阈值可以根据需要调整
            ##    print("距离过近！")
            ##    buzzer_on()
            ##    led_red_on()
            ##else:
            ##    buzzer_off()
            ##    led_red_off()

            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        # cap.release()
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
