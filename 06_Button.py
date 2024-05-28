import RPi.GPIO as GPIO

ii = 0

makerobo_BtnPin = 11  # 按钮接脚设置为11号
makerobo_Rpin = 12  # 红色LED接脚设置为12号
makerobo_Gpin = 13  # 绿色LED接脚设置为13号

# 初始化GPIO口
def makerobo_setup():
    GPIO.setmode(GPIO.BOARD)  # 采用实际的物理管脚给GPIO口
    GPIO.setwarnings(False)  # 关闭GPIO警告提示
    GPIO.setup(makerobo_Rpin, GPIO.OUT)  # 设置红色LED管脚为输出模式
    GPIO.setup(makerobo_Gpin, GPIO.OUT)  # 设置绿色LED管脚为输出模式
    GPIO.setup(makerobo_BtnPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # 设置按钮Pin管脚为输入模式，上拉至高电平
    GPIO.add_event_detect(makerobo_BtnPin, GPIO.BOTH, callback=makerobo_detect, bouncetime=200)  # 监听按钮Pin的状态变化

# 双色LED控制函数
def double_colorLED(x):
    if x == 0:  # x为0时，开启红色LED灯
        GPIO.output(makerobo_Rpin, 1)
        GPIO.output(makerobo_Gpin, 0)
    if x == 1:  # x为1时，开启绿色LED灯
        GPIO.output(makerobo_Rpin, 0)
        GPIO.output(makerobo_Gpin, 1)

# 打印函数，显示打印按钮按下
def makerobo_Print(x):
    if x == 0:
        ii = ii+1
        print(ii)
        print('****************************')
        print('*    Makerobo Kit Button Pressed!    *')
        print('****************************')

# 中断回调，有按钮按下时，响应中断函数
def makerobo_detect(chn):
    double_colorLED(GPIO.input(makerobo_BtnPin))  # 调用双色LED控制函数
    makerobo_Print(GPIO.input(makerobo_BtnPin))  # 打印出GPIO的状态值

# 循环函数
def makerobo_loop():
    while True:
        pass

# 销毁函数
def makerobo_destroy():
    GPIO.output(makerobo_Gpin, GPIO.LOW)  # 关闭绿色LED
    GPIO.output(makerobo_Rpin, GPIO.LOW)  # 关闭红色LED
    GPIO.cleanup()  # 清理资源

# 程序入口
if __name__ == '__main__':
    makerobo_setup()  # 初始化GPIO口
    try:
        makerobo_loop()  # 循环函数
    except KeyboardInterrupt:
        makerobo_destroy()  # 当按下Ctrl+C时, 执行destroy()子程序。
        # 清理资源
