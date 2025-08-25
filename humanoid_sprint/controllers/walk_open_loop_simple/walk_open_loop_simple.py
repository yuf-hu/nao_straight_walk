from controller import Supervisor
import math
import torch

class Sprinter(Supervisor):
    def __init__(self):
        super().__init__()
        self.timeStep = 40
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self):
        self.RShoulderPitch = self.getDevice('RShoulderPitch')
        self.LShoulderPitch = self.getDevice('LShoulderPitch')
        self.RShoulderPitch.setPosition(1.1)
        self.LShoulderPitch.setPosition(1.1)

        self.joint_names = [
            'RHipYawPitch', 'LHipYawPitch',
            'RHipRoll', 'LHipRoll',
            'RHipPitch', 'LHipPitch',
            'RKneePitch', 'LKneePitch',
            'RAnklePitch', 'LAnklePitch',
            'RAnkleRoll', 'LAnkleRoll'
        ]
        self.motors = {name: self.getDevice(name) for name in self.joint_names}
        self.sensors = {}
        for name in self.joint_names:
            sensor_name = name + "S"
            sensor = self.getDevice(sensor_name)
            sensor.enable(self.timeStep)
            self.sensors[sensor_name] = sensor
            setattr(self, sensor_name.replace("S", "Sensor"), sensor)

        self.accelerometer = self.getDevice("accelerometer")
        self.imu = self.getDevice("inertial unit")
        self.gyro = self.getDevice("gyro")
        self.accelerometer.enable(self.timeStep)
        self.imu.enable(self.timeStep)
        self.gyro.enable(self.timeStep)

        for _ in range(5):
            self.step(self.timeStep)

        self.nao = self.getFromDef("NAO")
        self.translation = self.nao.getField("translation")
        self.rotation = self.nao.getField("rotation")

        self.start_pos = self.get_translation()
        self.start_time = self.getTime()

    def get_translation(self):
        return self.translation.getSFVec3f()

    def get_yaw(self):
        return self.rotation.getSFRotation()[3]

    def left(self, x, y, z):
        self.motors['LKneePitch'].setPosition(z)
        self.motors['LHipPitch'].setPosition(-z/2 + x)
        self.motors['LAnklePitch'].setPosition(-z/2 - x)
        self.motors['LHipRoll'].setPosition(y)
        self.motors['LAnkleRoll'].setPosition(-y)

    def right(self, x, y, z):
        self.motors['RKneePitch'].setPosition(z)
        self.motors['RHipPitch'].setPosition(-z/2 + x)
        self.motors['RAnklePitch'].setPosition(-z/2 - x)
        self.motors['RHipRoll'].setPosition(y)
        self.motors['RAnkleRoll'].setPosition(-y)

    def run(self, target_x=4.5):
        f = 4
        robot_height = 0.5
        shift_y = 0.3
        step_height = 0.4
        step_length = 0.2
        arm_swing = 2.0

        steps = 0

        while self.step(self.timeStep) != -1:
            t = self.getTime() * f

            yLeftRight = math.sin(t) * shift_y
            zLeft  = (math.sin(t) + 1.0) / 2.0 * step_height + robot_height
            zRight = (math.sin(t + math.pi) + 1.0) / 2.0 * step_height + robot_height
            xLeft  = math.cos(t) * step_length
            xRight = math.cos(t + math.pi) * step_length

            self.left(xLeft, yLeftRight, zLeft)
            self.right(xRight, yLeftRight, zRight)

            self.RShoulderPitch.setPosition(arm_swing * xLeft + math.pi / 2 - 0.1)
            self.LShoulderPitch.setPosition(arm_swing * xRight + math.pi / 2 - 0.1)

            steps += 1
            pos = self.get_translation()
            if pos[2] < 0.2:
                print("Fell. Disqualified.")
                break
            if pos[0]  >= target_x:
                elapsed = self.getTime() - self.start_time
                print(f"Reached target x = {target_x:.2f} m in {elapsed:.2f} s over {steps} steps")
                break

        print("Done.")

if __name__ == '__main__':
    controller = Sprinter()
    controller.initialize()
    controller.run()
