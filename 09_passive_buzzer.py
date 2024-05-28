import RPi.GPIO as GPIO
import time

makerobo_Buzzer = 11  # 定义蜂鸣器管脚为11号

# 音调定义
CL = [0, 131, 147, 165, 175, 196, 211, 248]  # 低C音阶的频率
CM = [0, 262, 294, 330, 350, 393, 441, 495]  # 中C音阶的频率
CH = [0, 525, 589, 661, 700, 786, 882, 990]  # 高C音阶的频率

# 第一首歌曲谱
makerobo_song_1  = [
    CL[0], CL[0], CM[5], CM[5], CM[6], CL[5], CH[1], CM[7], CM[5], CM[5], CM[6], CM[5], CH[2],
]
# 小星星的节拍，1表示1/4拍
makerobo_beat_1  = [
    1, 1, 0.75, 0.25, 1, 1, 1, 2, 0.5, 0.5, 1, 1, 1
]

# 第二首歌曲谱
makerobo_song_2 = [
    CM[1], CM[1], CM[1], CL[5], CM[3], CM[3],
    CM[3], CM[1], CM[4], CM[3], CM[5], CM[1],
    CM[3], CM[3], CM[4], CM[3], CM[2], CM[2],
    CM[1]
]

# 第2首歌的节拍，1表示1/8拍
makerobo_beat_2 = [
    1, 1, 2, 2, 1, 1, 2, 2,
    1, 1, 2, 2, 1, 1, 3, 1,
    1, 2, 2
]

# GPIO设置函数
def makerobo_setup():
    GPIO.setmode(GPIO.BOARD)  # 采用实际的物理管脚给GPIO口
    GPIO.setwarnings(False)  # 关闭GPIO警告提示
    GPIO.setup(makerobo_Buzzer, GPIO.OUT)  # 设置蜂鸣器管脚为输出模式
    global makerobo_Buzz
    makerobo_Buzz = GPIO.PWM(makerobo_Buzzer, 440)  # 设置蜂鸣器频率为440
    makerobo_Buzz.start(50)  # 开始蜂鸣器并设置初始占空比为50%

# 循环函数
def makerobo_loop():
    while True:
        # 播放第一首歌...
        for i in range(1, len(makerobo_song_1)):
            makerobo_Buzz.ChangeFrequency(makerobo_song_1[i])  # 设置蜂鸣器的频率
            time.sleep(makerobo_beat_1[i] * 1)  # 按节拍延时
            makerobo_Buzz.ChangeFrequency(0)  # 设置蜂鸣器的频率
            time.sleep(0.1)  # 按节拍延时
            
        time.sleep(1)  # 每首歌之间暂停
        
        # 播放第二首歌...
        for i in range(1, len(makerobo_song_2)):
            makerobo_Buzz.ChangeFrequency(makerobo_song_2[i])  # 设置蜂鸣器的频率
            time.sleep(makerobo_beat_2[i] * 0.5)  # 按节拍延时

# 清理资源函数
def makerobo_destroy():
    makerobo_Buzz.stop()  # 停止蜂鸣器
    GPIO.output(makerobo_Buzzer, 1)  # 将蜂鸣器管脚输出设置为高电平
    GPIO.cleanup()  # 释放资源

# 程序入口
if __name__ == '__main__':
    makerobo_setup()  # 初始化设置
    try:
        makerobo_loop()  # 循环函数
    except KeyboardInterrupt:
        makerobo_destroy()  # 当按下Ctrl+C时, 执行destroy()子程序。
