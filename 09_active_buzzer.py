import RPi.GPIO as GPIO
import time

makerobo_Buzzer = 11  # 蜂鸣器管脚定义

# GPIO端口配置
def makerobo_setup(pin):
    global makerobo_BuzzerPin
    makerobo_BuzzerPin = pin
    GPIO.setmode(GPIO.BOARD)  # 采用实际的物理管脚给GPIO口
    GPIO.setwarnings(False)  # 关闭GPIO警告提示
    GPIO.setup(makerobo_BuzzerPin, GPIO.OUT)  # 设置GPIO脚为输出
    GPIO.output(makerobo_BuzzerPin, GPIO.HIGH)  # 蜂鸣器管脚输出高电平，关闭蜂鸣器

# 打开蜂鸣器
def makerobo_buzzer_on():
    GPIO.output(makerobo_BuzzerPin, GPIO.LOW)  # 输出低电平使蜂鸣器发声，低电平使蜂鸣器响起

# 关闭蜂鸣器
def makerobo_buzzer_off():
    GPIO.output(makerobo_BuzzerPin, GPIO.HIGH)  # 输出高电平关闭蜂鸣器

# 蜂鸣器响应控制
def makerobo_beep(x):
    makerobo_buzzer_on()  # 打开蜂鸣器
    time.sleep(x)  # 延时时间
    makerobo_buzzer_off()  # 关闭蜂鸣器
    time.sleep(x)  # 延时时间

# 循环函数
def loop():
    while True:
        makerobo_beep(0.5)  # 蜂鸣器响应控制, 延时时间为500ms

# 终止函数
def destroy():
    GPIO.output(makerobo_BuzzerPin, GPIO.HIGH)  # 输出高电平关闭蜂鸣器
    GPIO.cleanup()  # 释放资源

# 程序入口
if __name__ == '__main__':
    makerobo_setup(makerobo_Buzzer)  # 初始化GPIO端口
    try:
        loop()  # 循环函数
    except KeyboardInterrupt:
        destroy()  # 当按下Ctrl+C时, 执行destroy()子程序.
        # 终止函数
