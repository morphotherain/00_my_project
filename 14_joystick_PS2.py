import PCF8591 as ADC
import time

def makerobo_setup():
    ADC.setup(0x48)
    global makerobo_state

def makerobo_direction():
    state = ['home', 'up', 'down', 'left', 'right', 'pressed']
    i = 0
    if ADC.read(0) <= 30:
        i = 1
    if ADC.read(0) >= 225:
        i = 2
    if ADC.read(1) >= 225:
        i = 4
    if ADC.read(1) <= 30:
        i = 3
    if ADC.read(2) == 0 and ADC.read(1) == 128 and ADC.read(0) > 100 and ADC.read(0) < 130:
        i = 5
    if ADC.read(0) - 125 < 15 and ADC.read(0) - 125 > -15 and ADC.read(1) - 125 < 15 and ADC.read(1) - 125 > -15 and ADC.read(2) == 255:
        i = 0
    return state[i]

# 循环检测
def makerobo_loop():
    makerobo_status = ''  # 初始赋值空值
    while True:
        makerobo_tmp = makerobo_direction()  # 调用方向判断函数
        if makerobo_tmp != None and makerobo_tmp != makerobo_status:  # 判断状态是否有变化并且不为空
            print(makerobo_tmp)  # 打印出方向值
            makerobo_status = makerobo_tmp  # 把当前状态赋值给状态变量，以防止下一次状态变化时判断错误

# 异常处理函数
def destroy():
    pass

# 程序入口
if __name__ == '__main__':
    makerobo_setup()  # 初始化
    try:
        makerobo_loop()  # 调用循环检测
    except KeyboardInterrupt:  # 捕获Ctrl+C中断
        destroy()  # 调用异常处理函数