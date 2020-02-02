import rospy
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

CERT = 2            # set value for uncertainty estimate initially
MAX_SCAN = 3        # set longest scan distance to avoint 'inf' scan msgs

class Kalman:
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        if B is None:
            self.B = 0
        else:
            self.B = B
        if Q is None:
            self.Q = np.eye(self.n) 
        else:
            self.Q = Q
        if R is None:
            self.R = np.eye(self.n) 
        else: 
            self.R = R
        if P is None:
            self.P = np.eye(self.n) 
        else: 
            self.P = P 
        if x0 is None: 
            self.x = np.zeros((self.n, 1))
        else:
            self.x = x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def correct(self, z):
        self.r = z - np.dot(self.H, self.x)
        # compute sensor covariance
        self.S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.r)
        self.P = self.P - (np.dot(np.dot(np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S)), self.H), self.P)) 

class Run:
    def __init__(self):
        rospy.init_node('kalman')
        self.vel_sub = rospy.Subscriber("cmd_vel", Twist, self.vel_callback)
        self.scan_sub = rospy.Subscriber("scan", LaserScan, self.scan_callback)
        self.pose_pub = rospy.Publisher("PoseWithCovarianceStamped", PoseWithCovarianceStamped, queue_size=0)
        self.start_time = rospy.Time.now().to_sec()
        self.uncert = np.array([[CERT]])        # init uncertainty
        self.pos = 0
        self.vel = 0
        self.vel_scan_pts = []
        rospy.sleep(1)

    def vel_callback(self, msg):
        self.vel = msg.linear.x

    def scan_callback(self, msg):
        if msg.ranges[0] <= MAX_SCAN:
            self.scan = 2-msg.ranges[0] 			# Scan or 2-scan ???????
            self.t = rospy.Time.now().to_sec()
            self.execute()
        else:
            pass

    def execute(self):
        self.cur_time = rospy.Time.now().to_sec()
        delta_t = self.start_time - self.t
        
        F = np.array([[1]])                 # State transfromation
        B = np.array([[delta_t]])                 # delta t
        Q = np.array([[1]])                 # uncertainty in motion
        R = np.array([[1]])                 # error (stand dev)
        H = np.array([[1]])                 # world frame transformation
        a = Kalman(F, B, H, Q, R, self.uncert, self.pos)    # initialize 
        a.predict(self.vel)                 # predict with vel of 'u'm/s
        a.correct(abs(self.scan))                # predict with sense of 'z'm moved
        self.uncert = a.P
        print(self.scan)
        self.pos = a.x[0][0]

        pose_msg = PoseWithCovarianceStamped()
        rate = rospy.Rate(10)
        pose_msg.header.frame_id = 'frame'
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.pose.position.x = self.pos
        pose_msg.pose.covariance[0] = a.S[0][0]
        self.pose_pub.publish(pose_msg)
        rate.sleep()

        # append points for plotting later
        self.vel_scan_pts.append(((self.t - self.start_time), abs(self.pos)))
 
class SubAndPlot:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/pose', PoseStamped, self.odom_callback)
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)
        self.start_time = rospy.Time.now().to_sec()
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.uncert = np.array([[CERT]])        # init uncertainty
        self.vel = 0
        self.odom_pts = []
        self.scan_pose_pts = []
        rospy.sleep(1)
   
    def odom_callback(self, msg):
        self.pos = msg.pose.position.x
        self.t = rospy.Time.now().to_sec()
        self.odom_pts.append(((self.t - self.start_time), self.pos))

    def vel_callback(self, msg):
        self.vel = msg.linear.x

    def scan_callback(self, msg):
        if msg.ranges[0] <= MAX_SCAN:
            self.scan = 2-msg.ranges[0]                         # Scan or 2-scan ???????
            self.time = rospy.Time.now().to_sec()
            self.run_kalman()
        else:
            pass

    def run_kalman(self):
        self.cur_time = rospy.Time.now().to_sec()
        delta_t = self.start_time - self.time

        F = np.array([[1]])                 # State transfromation
        B = np.array([[delta_t]])                 # delta t
        Q = np.array([[1]])                 # uncertainty in motion
        R = np.array([[1]])                 # error (stand dev)
        H = np.array([[1]])                 # world frame transformation
        a = Kalman(F, B, H, Q, R, self.uncert, self.pos)    # init
        a.predict(self.vel)                        # predict with vel of 'u'm/s
        a.correct(self.scan)        # predict with sense of 'z'm moved
        self.uncert = a.P
      #  self.pos = a.x[0][0]
        self.scan_pose_pts.append(((self.time - self.start_time), round(self.pos, 2)))

    def plot(self, pts, figname):
        fig = plt.figure()
        fig.suptitle(figname, fontsize=20)
        plt.xlabel("time in sec", fontsize=15)
        plt.ylabel("position in m", fontsize=15)
        plt.plot([pts[0][0]], [pts[0][1]], 'ro', label=figname)
        for i in pts:
            plt.plot([i[0]],[i[1]], 'ro')
        #plt.show()
        fig.savefig(figname+'.png')
        print("plotted")

if __name__ == '__main__':
    a = Run()
    b = SubAndPlot()
    rospy.sleep(18)
#    print(b.scan_pose_pts)
#    print(a.vel_scan_pts)
    b.plot(b.odom_pts, "new_odom2")
    b.plot(b.scan_pose_pts, "new_scan_pose")
    b.plot(a.vel_scan_pts, "new_vel_scan")
