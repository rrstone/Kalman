import rospy
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

BAG_DUR = 18        # duration of bag (used to sleep so all messages are recieved)
CERT = .1           # set value for uncertainty estimate initially
MAX_SCAN = 3        # set longest scan distance to avoint 'inf' scan msgs

# extracted start/end positions from csv manually since there were 2 values
START = 5.3         # start time of movement forward in bag (actual)
START_POS = 0       # initial position (actual)
END = 10.3           # end time of movement in bag
END_POS = 0.98      # final position (acutal)

# class to implement the Kalman Filter
class Kalman:
    #initialize matrices to correct sizes
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

    # function for propogation step
    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    # function for correction step
    def correct(self, z):
        self.r = z - np.dot(self.H, self.x)
        self.S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.K, self.r)
        self.P = self.P - (np.dot(np.dot(np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S)), self.H), self.P)) 

# class for publishing PoseWithCovarianceStamped msgs (from kalman estimates)
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

    # callback for cmd_vel subscriber
    def vel_callback(self, msg):
        self.vel = msg.linear.x

    # callback for scan subscriber
    def scan_callback(self, msg):
        if msg.ranges[0] <= MAX_SCAN:
            self.scan = 2-msg.ranges[0]        # put in global frame
            self.t = rospy.Time.now().to_sec()
            self.execute()
        else:
            pass

    # function to implement Kalman filter, publish messages, and store points for plotting
    def execute(self):
        self.cur_time = rospy.Time.now().to_sec()
        delta_t = self.start_time - self.t
        
        F = np.array([[1]])                 # State transfromation
        B = np.array([[delta_t]])                 # delta t
        Q = np.array([[1]])                 # uncertainty in motion
        R = np.array([[1]])                 # error (stand dev)
        H = np.array([[1]])                 # world frame transformation
        a = Kalman(F, B, H, Q, R, self.uncert, self.pos)    # initialize Kalman 
        a.predict(self.vel)                 # predict with vel of 'u'm/s
        a.correct(self.scan)                # predict with sense of 'z'm moved
        self.uncert = a.P                   # update P
        self.pos = a.x[0][0]                # update position

        # create and publish msg
        pose_msg = PoseWithCovarianceStamped()
        rate = rospy.Rate(10)
        pose_msg.header.frame_id = 'frame'
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.pose.position.x = self.pos
        pose_msg.pose.covariance[0] = a.S[0][0]
        self.pose_pub.publish(pose_msg)
        rate.sleep()

        # append points for plotting later
        self.vel_scan_pts.append(((self.t - self.start_time), self.pos))

# class for subscribing to topics and implementing Kalman filter, and plotting info
class SubAndPlot:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/pose', PoseStamped, self.odom_callback)
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)
        self.start_time = rospy.Time.now().to_sec()
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.uncert = np.array([[CERT]])        # init uncertainty
        self.vel = 0
        self.pos = 0
        self.odom_pts = []
        self.scan_pose_pts = []
        self.plot_state = 0
        self.fig = plt.figure()
        rospy.sleep(1)
   
    # callback for pose subscriber
    def odom_callback(self, msg):
        self.pos = msg.pose.position.x
        self.t = rospy.Time.now().to_sec()
        self.odom_pts.append(((self.t - self.start_time), self.pos))

    # callback for cmd_vel subscriber
    def vel_callback(self, msg):
        self.vel = msg.linear.x

    # callback for scan subscriber
    def scan_callback(self, msg):
        if msg.ranges[0] <= MAX_SCAN:
            self.scan = 2-msg.ranges[0]         # put in global frame
            self.time = rospy.Time.now().to_sec()
            self.run_kalman()
        else:
            pass

    # function for implementing Kalman filter and storing points for plotting
    def run_kalman(self):
        self.cur_time = rospy.Time.now().to_sec()
        delta_t = self.start_time - self.time
        F = np.array([[1]])                 # State transfromation
        B = np.array([[delta_t]])           # delta t
        Q = np.array([[1]])                 # uncertainty in motion
        R = np.array([[1]])                 # error (stand dev)
        H = np.array([[1]])                 # world frame transformation
        a = Kalman(F, B, H, Q, R, self.uncert, self.pos)    # init
        a.predict(self.vel)                        # predict with vel of 'u'm/s
        a.correct(self.scan)                # predict with sense of 'z'm moved
        self.uncert = a.P
        self.pos = a.x[0][0]		    # Updated pose estimate
        # store for plotting later
        self.scan_pose_pts.append(((self.time - self.start_time), round(self.pos, 2)))

    # function for plotting information from a list of tuples
    def plot(self, pts, figname, dot, err):
#        self.fig = plt.figure()
        self.fig.suptitle("Position vs Time", fontsize=20)
        plt.xlabel("time in sec", fontsize=15)
        plt.ylabel("position in m", fontsize=15)
        plt.plot([pts[0][0]], [pts[0][1]], dot, label=figname)
        for i in pts:
            plt.plot([i[0]],[i[1]], dot)
        last = len(pts)-1
        plt.plot([START], [START_POS - pts[0][1]], err, label='error')
        plt.plot([END], [END_POS - pts[last][1]], err)
        
        if self.plot_state == 0:
            plt.plot([START], [START_POS], 'bo', label='Actual pos')
            plt.plot([END], [END_POS], 'bo')
            self.plot_state = 1
        plt.legend(loc='center right')
        print("plotted")

if __name__ == '__main__':
    a = Run()
    b = SubAndPlot()
    rospy.sleep(BAG_DUR)
    b.plot(b.odom_pts, "Pose", 'ro', 'rx')
    b.plot(b.scan_pose_pts, "Scan and Pose", 'go', 'gx')
    b.plot(a.vel_scan_pts, "Vel and Scan", 'mo', 'mx')
    b.fig.savefig("plot.png")

