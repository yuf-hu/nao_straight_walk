from controller import Robot
import math

class Sprinter(Robot):
    def initialize(self):
        self.timeStep = int(self.getBasicTimeStep())

        # Arm motors
        self.RShoulderPitch = self.getMotor('RShoulderPitch')
        self.LShoulderPitch = self.getMotor('LShoulderPitch')

        # Leg motors
        self.RHipYawPitch = self.getMotor('RHipYawPitch')
        self.LHipYawPitch = self.getMotor('LHipYawPitch')
        self.RHipRoll     = self.getMotor('RHipRoll')
        self.LHipRoll     = self.getMotor('LHipRoll')
        self.RHipPitch    = self.getMotor('RHipPitch')
        self.LHipPitch    = self.getMotor('LHipPitch')
        self.RKneePitch   = self.getMotor('RKneePitch')
        self.LKneePitch   = self.getMotor('LKneePitch')
        self.RAnklePitch  = self.getMotor('RAnklePitch')
        self.LAnklePitch  = self.getMotor('LAnklePitch')
        self.RAnkleRoll   = self.getMotor('RAnkleRoll')
        self.LAnkleRoll   = self.getMotor('LAnkleRoll')

        # 记录初始时间和起始位置
        self.start_time = self.getTime()
        self.start_x = self.getSelf().getField("translation").getSFVec3f()[0]

    def left(self, x, y, z):
        self.LKneePitch.setPosition ( z      )
        self.LHipPitch.setPosition  (-z/2 + x)
        self.LAnklePitch.setPosition(-z/2 - x)
        self.LHipRoll.setPosition( y)
        self.LAnkleRoll.setPosition(-y)

    def right(self, x, y, z):
        self.RKneePitch.setPosition ( z      )
        self.RHipPitch.setPosition  (-z/2 + x)
        self.RAnklePitch.setPosition(-z/2 - x)
        self.RHipRoll.setPosition( y)
        self.RAnkleRoll.setPosition(-y)

    def run(self):
        # === 步态参数（你可切换其他配置） ===
        f            = 4
        robot_height = 0.5
        shift_y      = 0.3
        step_height  = 0.4
        step_length  = 0.2
        arm_swing    = 2.0 

        # === 主循环 ===
        while self.step(self.timeStep) != -1:
            t = self.getTime() * f

            y = math.sin(t) * shift_y
            zLeft  = (math.sin(t)           + 1.0) / 2.0 * step_height + robot_height
            zRight = (math.sin(t + math.pi) + 1.0) / 2.0 * step_height + robot_height
            xLeft  = math.cos(t) * step_length
            xRight = math.cos(t + math.pi) * step_length

            self.left( xLeft, y, zLeft )
            self.right(xRight, y, zRight )

            self.RShoulderPitch.setPosition(arm_swing * xLeft  + math.pi/2 - 0.1)
            self.LShoulderPitch.setPosition(arm_swing * xRight + math.pi/2 - 0.1)

            # === 实时检查是否到达目标或跌倒 ===
            pos = self.getSelf().getField("translation").getSFVec3f()
            cur_x = pos[0]
            cur_z = pos[2]  # Webots 的 z 是高度（竖直方向）

            if cur_z < 0.2:
                print("💥 Robot fell down! Disqualified.")
                break

            if cur_x - self.start_x >= 4.5:
                elapsed = self.getTime() - self.start_time
                print(f"✅ Reached 4.5 m in {elapsed:.2f} s!")
                break

# === 启动控制器 ===
controller = Sprinter()
controller.initialize()
controller.run()