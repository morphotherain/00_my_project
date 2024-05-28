import RPi.GPIO as GPIO
import time

colors = [0xFF00, 0x00FF, 0x0FF0, 0xF00F]  # 定义颜色数组
makerobo_pins = (11, 12)  # PIN脚编号

GPIO.setmode(GPIO.BOARD)  # 板载模式的编号设置用于GPIO
GPIO.setwarnings(False)  # 关闭GPIO警告
GPIO.setup(makerobo_pins, GPIO.OUT)  # 设置Pin脚为输出模式以控制LED
GPIO.output(makerobo_pins, GPIO.LOW)  # 设置Pin脚的输出电平为0V关闭LED

P_R = GPIO.PWM(makerobo_pins[0], 2000)  # 设置红色频率为2KHz
P_G = GPIO.PWM(makerobo_pins[1], 2000)  # 设置绿色频率为2KHz
P_R.start(0)
P_G.start(0)

def makerobo_pwm_map(x, in_min, in_max, out_min, out_max):
    # 将输入的值从一个范围映射到另一个范围
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def makerobo_set_Color(col):
    # 制造: col = 0x1122
    R_val = col >> 8  # 0~255的范围可以将信号0-100之间
    G_val = col & 0x00FF  # 0~255的范围可以将信号0-100之间
    R_val = makerobo_pwm_map(R_val, 0, 255, 0, 100)
    G_val = makerobo_pwm_map(G_val, 0, 255, 0, 100)
    P_R.ChangeDutyCycle(R_val)  # 改变占空比
    P_G.ChangeDutyCycle(G_val)  # 改变占空比

def makerobo_loop():
    # 闪烁循环函数
    while True:
        for col in colors:
            makerobo_set_Color(col)
            time.sleep(0.5)  # 间隔时间

def makerobo_destroy():
    # 程序终止清理
    P_G.stop()
    P_R.stop()
    GPIO.output(makerobo_pins, GPIO.LOW)  # 关闭所有LED
    GPIO.cleanup()

if __name__ == "__main__":
    try:
        makerobo_loop()
    except KeyboardInterrupt:  # 当按下Ctrl+C时，清理destroy()函数
        makerobo_destroy()
