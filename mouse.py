import smbus
import time
import math
from pynput.mouse import Controller as MouseController

# MPU6050定义
MPU6050_ADDR = 0x68
POWER_MGMT_1 = 0x6b
POWER_MGMT_2 = 0x6c

# 初始化SMBus
bus = smbus.SMBus(1)

# 初始化pynput鼠标控制器
mouse = MouseController()

# 前一次测量的角度
previous_x = 0
previous_y = 0
# 鼠标移动灵敏度
sensitivity = 10
# 角度变化阈值
threshold = 6  # 角度

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

def read_word_2c(adr):
    val = (bus.read_byte_data(MPU6050_ADDR, adr) << 8) + bus.read_byte_data(MPU6050_ADDR, adr + 1)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val

def read_mpu6050():
    accel_xout = read_word_2c(0x3b)
    accel_yout = read_word_2c(0x3d)
    accel_zout = read_word_2c(0x3f)
    accel_xout_scaled = accel_xout / 16384.0
    accel_yout_scaled = accel_yout / 16384.0
    accel_zout_scaled = accel_zout / 16384.0

    x = get_x_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)
    y = get_y_rotation(accel_xout_scaled, accel_yout_scaled, accel_zout_scaled)
    
    return (x, y)

def move_mouse_based_on_tilt():
    global previous_x, previous_y

    while True:
        x, y = read_mpu6050()  # 读取当前的倾斜角度
        # 计算角度变化
        delta_x = x - previous_x
        delta_y = y - previous_y
        # 更新上一次的角度
        previous_x = x
        previous_y = y
        # 检查角度变化是否超过误差限
        if abs(delta_x) > threshold or abs(delta_y) > threshold:
            # 计算鼠标应移动的距离
            move_x = -int(delta_x * sensitivity)
            move_y = int(delta_y * sensitivity)
            # 移动鼠标
            mouse.move(move_x, move_y)

        # 等待一定时间后再次检测
        time.sleep(0.1)  # 检测间隔，可以根据需要调整

if __name__ == '__main__':
    init_mpu6050()
    move_mouse_based_on_tilt()
