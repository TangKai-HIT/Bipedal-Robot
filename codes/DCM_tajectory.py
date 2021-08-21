'''
core functions and classes for trajectory planning of 
bipedal robot's divergent  component of motion(DCM)
author:TangKai
reference:Johannes Englsberger, Christian Ott, and Alin Albu-Schäffer. Three-dimensional bipedal walking 
        control using divergent component of mo-tion. In2013 IEEE/RSJ International Conference on 
        Intelligent Robotsand Systems, pages 2600–2607. IEEE.
'''
import numpy as np
import math
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
#import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
#import mpl_toolkits.mplot3d.art3d as art3d

class bipalRobot(object):
    #constructor
    def __init__(self, m) -> None:
        # m:mass; x:center of mass;
        # footPs:foot points
        self.mass = m #mass(kg)
        self.footPs = [] #reference foot points location(list), unit:m
        self.delz_vrp = None #z-axis offset from eCMP
        self.DCM_Traj = [] #planned DCM trajectory
        self.COM_Traj = [] #planned COM trajectory
        self.timePS_SS = [] #time for one step during single support 
        self.sampTime = 0.05 #sampling time gap
        self.timeSqu = [] #time sequence of the trajectory
        self.sampNum = 0 #sampled points' number

    #DCM planner for single support case
    def DCM_PlannerSS(self, initCOM, footPs, timePS_SS, sampTime=0.01, plot=True, savefig=True) -> None:
        #Step up VRP and foot points
        self.DCM_Traj=[] #clear existing DCM trajectory
        self.COM_Traj=[] #clear existing COM trajectory
        g=9.8
        self.footPs = footPs
        self.COM_Traj.append(np.array(initCOM))
        self.delz_vrp = initCOM[-1] - footPs[-1][-1] #walking stops at the last step(COM,VRP and DCM coincide)
        a = math.sqrt(g/self.delz_vrp) #A constant for DCM calculation
        self.timePS_SS = timePS_SS
        self.sampTime = sampTime

        #offset from foot points to construct VRPs
        r_VRP = list(map(lambda x: np.array(x)+[0,0,self.delz_vrp], footPs))

        #Backward recursion
        DCM_key = [] #DCM key points
        for i in range(len(r_VRP)):
            if i==0: 
                DCM_key.append(r_VRP[-1-i]) #recursion start from the end
            else:
                newDCM_key = r_VRP[-1-i] + math.exp(-a*self.timePS_SS)*(DCM_key[i-1]-r_VRP[-1-i])
                DCM_key.append(newDCM_key)
        DCM_key.reverse() #reverse the backward key points sequence to be forward

        #Forward step sampling, generate DCM trajetory and COM trajectory
        #define DCM trajectory equation and COM function for solving COM dynamics
        def COM_DCM_equ(r_vrp, DCM_ini):
             DCM_equ = lambda t: r_vrp + math.exp(a*t) * (DCM_ini - r_vrp)
             COM_func = lambda t, x: -a*(x - DCM_equ(t))
             return DCM_equ, COM_func

        pointNumPS =  timePS_SS/sampTime #number of points per step 
        for i in range(len(DCM_key)):
            self.timeSqu.append(i*timePS_SS) #insert key points timing
            self.DCM_Traj.append(DCM_key[i]) #insert key points

            if i != len(DCM_key)-1: #calculate DCM and COM(RK45 solver) until DCM reach the end 
                DCM_equ, COM_func = COM_DCM_equ(r_VRP[i], DCM_key[i]) #reconstruct lambda function
                #assign DCM tarjectory points and time squence
                for k in range(1,int(pointNumPS)):
                    t_dcm = k*sampTime
                    t = i*timePS_SS + t_dcm
                    self.timeSqu.append(t)
                    self.DCM_Traj.append(DCM_equ(t_dcm))
                #solve the COM dynamics equation
                #X0 = self.COM_Traj[int(i*pointNumPS)]
                X0 = self.COM_Traj[-1]
                t_eval0 = np.linspace(0,timePS_SS,int(pointNumPS+1))
                sol = solve_ivp(COM_func, [0,timePS_SS], X0, method='RK45', t_eval=t_eval0) 
                X = sol.y; X = X.transpose()
                self.COM_Traj += list(X[1:])
            else: #continue to calculate COM until it coincides with the last DCM point
                COM_func = lambda x: -a*(x - DCM_key[i]) #construct lambda function
                COM_now = self.COM_Traj[-1]
                while np.linalg.norm((COM_now - DCM_key[i]), ord=2) > 1e-4:
                    self.timeSqu.append(self.timeSqu[-1]+sampTime)

                    f_now = COM_func(self.COM_Traj[-1])
                    COM_new = self.COM_Traj[-1] + sampTime * COM_func(COM_now + (sampTime/2)*f_now) #modified Eular format(explicit)

                    self.COM_Traj.append(COM_new)
                    COM_now = COM_new

        #plot trajectory in 3D
        if plot:
            plt.figure('single support trajectory',clear=True)
            fig = plt.figure('single support trajectory',figsize=(100,100))
            ax = fig.add_subplot(projection='3d')
            plt.cla()
            ax.plot([p[0] for p in self.DCM_Traj],
                    [p[1] for p in self.DCM_Traj],
                    [p[2] for p in self.DCM_Traj], color='blue', linewidth=1, label="DCM") #plot DCM trajectory
            ax.plot([p[0] for p in self.COM_Traj],
                    [p[1] for p in self.COM_Traj],
                    [p[2] for p in self.COM_Traj], color='green', linewidth=1, label="COM") #plot COM trajectory
            plt.legend(loc='upper right')

            ax.scatter([p[0] for p in r_VRP],
                        [p[1] for p in r_VRP],
                        [p[2] for p in r_VRP], color='black') #VRP points
            ax.scatter([p[0] for p in footPs],
                        [p[1] for p in footPs],
                        [p[2] for p in footPs], color='red') #foot points
            ax.scatter([p[0] for p in DCM_key],
                        [p[1] for p in DCM_key],
                        [p[2] for p in DCM_key], color='blue') #DCM key points

            ax.set_xlabel('X(m)'); ax.set_ylabel('Y(m)'); ax.set_zlabel('Z(m)') #axis label

            lenth=np.abs(footPs[-1][1]-footPs[0][1]); 
            plt.xlim([-lenth/2,lenth/2]); plt.ylim([footPs[0][1],footPs[-1][1]]) #axis scale

            ax.set_title('Robot Trajetory(DCM,COM) [$t_{step,i}=%.2f(s)$]' % timePS_SS) #set title

            text = 'total time=%.2f(s)' % (self.timeSqu[-1])
            ax.text2D(0.8, 0, text, color='red', transform=ax.transAxes) #text: total time

            #annote initial COM point and final COM point
            ax.scatter(initCOM[0],initCOM[1],initCOM[2], color='green')
            x0, y0, _ = proj3d.proj_transform(initCOM[0],initCOM[1],initCOM[2], ax.get_proj())
            plt.annotate(
                    r"$\mathbf{x}_{ini}$", xy = (x0, y0), xytext = (20, -40),
                    textcoords = 'offset points', ha = 'left', va = 'top',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
            
            finalCOM = self.COM_Traj[-1]
            ax.scatter(finalCOM[0],finalCOM[1],finalCOM[2], color='green')
            xf, yf, _ = proj3d.proj_transform(finalCOM[0],finalCOM[1],finalCOM[2], ax.get_proj())
            plt.annotate(
                    r"$\mathbf{x}_{end}$", xy = (xf, yf), xytext = (20, -40),
                    textcoords = 'offset points', ha = 'left', va = 'top',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))

            #annotate foot points and VRP, plot lines connecting VRP and foot points
            for i in range(len(footPs)): 
                f_point = footPs[i]; vrp_point = r_VRP[i]
                label1 = "$\\mathbf{r}_{f,%d}$" % (i+1)
                label2 = "$\\mathbf{r}_{vrp,%d}$" % (i+1)
                x1, y1, _ = proj3d.proj_transform(f_point[0],f_point[1],f_point[2], ax.get_proj())
                x2, y2, _ = proj3d.proj_transform(vrp_point[0],vrp_point[1],vrp_point[2], ax.get_proj())
                plt.annotate(
                    label1, xy = (x1, y1), xytext = (-20, 20),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
                plt.annotate(
                    label2, xy = (x2, y2), xytext = (-20, 20),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
                ax.plot([f_point[0],vrp_point[0]], [f_point[1],vrp_point[1]], 
                        [f_point[2],vrp_point[2]], color='black', linewidth=1)

            #annotate DCM key points
            for i in range(len(DCM_key)): 
                point = DCM_key[i]; label = "$\\mathbf{\\xi}_{d,%d}$" % (i+1)
                x1, y1, _ = proj3d.proj_transform(point[0],point[1],point[2], ax.get_proj())
                plt.annotate(
                label, xy = (x1, y1), xytext = (10, -30),
                textcoords = 'offset points', ha = 'left', va = 'top',
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
            
            if savefig:  
                plt.savefig('DCM_SS.eps',format='eps')
                plt.savefig('DCM_SS.jpg',format='jpg')
            plt.show()
            


if __name__=="__main__":
    robot = bipalRobot(80)
    footSteps = [[-0.2, 0.0, 0.0],
                [0.2, 0.8, 0.0],
                [-0.2, 0.8*2, 0.0],
                [0.2, 0.8*3, 0.0],
                [-0.2, 0.8*4, 0.0],
                [0.2, 0.8*5, 0.0]]
    initialCOM = [0.0, 0.0, 1]
    robot.DCM_PlannerSS(initialCOM, footSteps, 0.6) #single support test