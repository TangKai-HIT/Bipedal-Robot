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
from numpy.lib.function_base import append
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
#import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
#import mpl_toolkits.mplot3d.art3d as art3d

class bipalRobot(object):
    #constructor
    def __init__(self, m, footSize) -> None:
        # m:mass; x:center of mass;
        # footPs:foot points
        self.mass = m #mass(kg)
        self.footSize = footSize #foot size:[length, width](rectangle box)
        self.footPs = [] #reference foot points location(list), unit:m
        self.footRot = 0 #rotation of foot in each foot points,unit:rad
        self.delz_vrp = None #z-axis offset from eCMP
        self.DCM_Traj = [] #planned DCM trajectory
        self.COM_Traj = [] #planned COM trajectory
        self.timePS_SS = None #single support time per step
        self.timePS_DS = None #double support time per step
        self.sampTime = None #sampling time gap
        self.timeSqu = [] #time sequence of the trajectory
    
    #protected methods: 
    #A tool for the DCM_planners(solve dynamics equation in a motion segment and add points to trajectory)
    def _addTrajRK45(self, time, function) -> None:
        X0 = self.COM_Traj[-1]
        t_eval0 = np.linspace(0,time,int(time/self.sampTime+1))
        sol = solve_ivp(function, [0,time], X0, method='RK45', t_eval=t_eval0) 
        X = sol.y; X = X.transpose()
        self.COM_Traj += list(X[1:])
    
    #define single support DCM trajectory equation and COM function for solving COM dynamics
    def _COM_DCM_equSS(self, r_vrp, DCM_ini, a):
        DCM_equ = lambda t: r_vrp + math.exp(a*t) * (DCM_ini - r_vrp)
        COM_func = lambda t, x: -a*(x - DCM_equ(t))
        return DCM_equ, COM_func
    
    #define double support DCM trajectory equation and its derivative and COM function for solving COM dynamics
    def _COM_DCM_equDS(self, DCM_DSini, DCM_DSend, dDCM_DSini, dDCM_DSend, a, time):
        #set up polynomial parameter matrix for double support DCM interpolation
        P = np.array([[2/(time**3), 1/(time**2), -2/(time**3), 1/(time**2)],
                        [-3/(time**2), -2/time, 3/(time**2), -1/time],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0]])
        bound = np.array([DCM_DSini, dDCM_DSini, DCM_DSend, dDCM_DSend])
        interP = np.matmul(P, bound) 
        DCM_equ = lambda t: np.matmul([t**3,t**2,t,1], interP)
        dDCM_equ = lambda t: np.matmul([3*(t**2),2*t,1,0], interP)
        COM_func = lambda t, x: -a*(x - DCM_equ(t))
        return DCM_equ, dDCM_equ, COM_func

    #DCM planner for single support cases
    def DCM_PlannerSS(self, initCOM, footPs, timePS_SS, sampTime=0.01, plot=True, savefig=True) -> None:
        #Step up VRP and foot points
        self.DCM_Traj=[] #clear existing DCM trajectory
        self.COM_Traj=[] #clear existing COM trajectory
        self.timeSqu = [] #clear existing time sequence
        g=9.8
        self.footPs = footPs
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
        pointNumPS =  timePS_SS/sampTime #number of points per step 
        self.COM_Traj.append(np.array(initCOM))
        for i in range(len(DCM_key)):
            self.timeSqu.append(i*timePS_SS) #insert key points timing
            self.DCM_Traj.append(DCM_key[i]) #insert key points

            if i != len(DCM_key)-1: #calculate DCM and COM(RK45 solver) until DCM reach the end 
                DCM_equ, COM_func = self._COM_DCM_equSS(r_VRP[i], DCM_key[i], a) #reconstruct lambda function
                #assign DCM tarjectory points and time squence
                for k in range(1,int(pointNumPS)):
                    t_dcm = k*sampTime
                    t = i*timePS_SS + t_dcm
                    self.timeSqu.append(t)
                    self.DCM_Traj.append(DCM_equ(t_dcm))
                #solve the COM dynamics equation
                self._addTrajRK45(timePS_SS,COM_func)
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
            ax.view_init(elev=60, azim=-60) #adjust view
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

            ax.set_title('Robot Trajetory(DCM,COM)(Single Support)') #set title

            text = '$t_{step,i}=%.2f(s)$\ntotal time=%.2f(s)' % (timePS_SS, self.timeSqu[-1])
            ax.text2D(0.8, 0, text, color='red', transform=ax.transAxes) #text: single support time per step & total time

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

    #DCM planner for double support cases
    def DCM_PlannerDS(self, initCOM, footPs, timePS_SS, timePS_DS, DS_ratio=0.5, sampTime=0.01, plot=True, savefig=True) -> None:        
        #Step up basic parameters
        self.DCM_Traj=[] #clear existing DCM trajectory
        self.COM_Traj=[] #clear existing COM trajectory
        self.timeSqu = [] #clear existing time sequence
        g=9.8
        self.footPs = footPs
        self.delz_vrp = initCOM[-1] - footPs[-1][-1] #walking stops at the last step(COM,VRP and DCM coincide)
        a = math.sqrt(g/self.delz_vrp) #A constant for DCM calculation
        self.timePS_SS = timePS_SS
        self.timePS_DS = timePS_DS 
        self.sampTime = sampTime
        t_iniDS = DS_ratio*timePS_DS #initial double support time(from the DCM key points in single support)
        t_endDS = timePS_DS - t_iniDS #end double support time(from the DCM key points in single support)

        #offset from foot points to construct VRPs
        r_VRP = list(map(lambda x: np.array(x)+[0,0,self.delz_vrp], footPs))

        #Backward recursion
        DCM_key = [] #DCM key points
        dDCM_key = [] #DCM key points' derivatives
        DCM_iniDS = [] #DCM initial double support points
        dDCM_iniDS = [] #1-order derivative of DCM initial double support points
        DCM_endDS = [] #DCM end double support points
        dDCM_endDS = [] #1-order derivative of DCM end double support points
        for i in range(len(r_VRP)):
            if i==0: 
                DCM_key.append(r_VRP[-1]) #recursion start from the end DCM key
                newDCM_iniDS = r_VRP[-2] + math.exp(-a*t_iniDS)*(DCM_key[-1]-r_VRP[-2]) #calculate nearby initial DS key
                DCM_iniDS.append(newDCM_iniDS) #add new initial DS key point to sequence
                DCM_endDS.append(r_VRP[-1]) #the last end DS key point is the last DCM key

                newdDCM_iniDS = a*math.exp(-a*t_iniDS)*(DCM_key[-1]-r_VRP[-2]) #calculate the derivative of nearby initial DS key
                dDCM_iniDS.append(newdDCM_iniDS) 
                dDCM_endDS.append(np.array([0,0,0])) #the derivative of the last end DS key point is 0
            else:
                newDCM_key = r_VRP[-1-i] + math.exp(-a*self.timePS_SS)*(DCM_key[i-1]-r_VRP[-1-i]) #calculate new DCM key
                DCM_key.append(newDCM_key) #add new DCM key point to sequence
                newDCM_endDS = r_VRP[-1-i] + math.exp(a*t_endDS)*(newDCM_key-r_VRP[-1-i]) #calculate nearby end DS key
                DCM_endDS.append(newDCM_endDS) #add new end DS key point to sequence
                newdDCM_endDS = a*math.exp(a*t_endDS)*(newDCM_key-r_VRP[-1-i]) #calculate the derivative of nearby end DS key
                dDCM_endDS.append(newdDCM_endDS) #add it to sequence
                
                if i<(len(r_VRP)-1): #initial DS key point in normal condition
                    newDCM_iniDS = r_VRP[-2-i] + math.exp(-a*t_iniDS)*(newDCM_key-r_VRP[-2-i]) #calculate nearby initial DS key
                    DCM_iniDS.append(newDCM_iniDS) #add new initial DS key point to sequence
                    newdDCM_iniDS = a*math.exp(-a*t_iniDS)*(newDCM_key-r_VRP[-2-i]) #calculate nearby initial DS key
                    dDCM_iniDS.append(newdDCM_iniDS) 
                else: #the first initial DS key point
                    DCM_iniDS.append(newDCM_key) #the first initial DS key point is the first DCM key
                    newdDCM_iniDS = a*(newDCM_key - r_VRP[-1-i])
                    dDCM_iniDS.append(newdDCM_iniDS) 

        DCM_keySS = DCM_key #store the DCM SS key points
        DCM_key = [] #reset DCM key points
        DCM_iniDS.reverse() #reverse the backward DCM DS initial points sequence to be forward
        DCM_endDS.reverse() #reverse the backward DCM DS end points sequence to be forward
        dDCM_iniDS.reverse() #reverse the backward DCM DS initial points' derivatives sequence to be forward
        dDCM_endDS.reverse() #reverse the backward DCM DS end points' derivatives sequence to be forward

        #Rearrange DCM DS initial points and end points into one squence
        for i in range(len(DCM_iniDS)):
            DCM_key.append(DCM_iniDS[i]); DCM_key.append(DCM_endDS[i])
            dDCM_key.append(dDCM_iniDS[i]); dDCM_key.append(dDCM_endDS[i])
        del DCM_iniDS, DCM_endDS, dDCM_iniDS, dDCM_endDS #delete unuseful variables

        #Forward step sampling, generate DCM trajetory and COM trajectory
        VRP_Traj = [] #VRP trajectory
        self.timeSqu.append(0) #start time sequence
        self.COM_Traj.append(np.array(initCOM)) #insert the first COM point
        for i in range(len(DCM_key)):
            DCM_equ=[]; COM_func=[]; dDCM_equ=[]; duration=None #reset variables
            self.DCM_Traj.append(DCM_key[i]) #insert DCM key point
            if i < len(DCM_key)-1:  #calculate DCM and COM(RK45 solver) until DCM reach the end   
                if not(i%2): #index of initDS points
                    if i==0: #first initDS point
                        duration = t_endDS
                        DCM_equ, COM_func = self._COM_DCM_equSS(r_VRP[math.floor(i/2)], DCM_key[i], a) #reconstruct lambda function
                    else:
                        duration = timePS_DS if i<len(DCM_key)-2 else t_iniDS
                        DCM_equ, dDCM_equ, COM_func = self._COM_DCM_equDS(DCM_key[i], DCM_key[i+1], dDCM_key[i], 
                                                                    dDCM_key[i+1], a, duration) #reconstruct lambda function
                else: #index of endDS points
                    duration = timePS_SS - timePS_DS
                    DCM_equ, COM_func = self._COM_DCM_equSS(r_VRP[math.floor(i/2)], DCM_key[i], a) #reconstruct lambda function

                for k in range(1,int(duration/sampTime)):
                    t_dcm = k*sampTime
                    self.timeSqu.append(self.timeSqu[-1] + sampTime) #update time sequences
                    self.DCM_Traj.append(DCM_equ(t_dcm))
                    if not(i%2) and i!=0 and i!=len(DCM_key)-2:
                        VRP_Traj.append(DCM_equ(t_dcm) - (1/a)*dDCM_equ(t_dcm))
                    elif i==0:
                        VRP_Traj.append(r_VRP[i]) #insert VRP key point
                    else:
                        VRP_Traj.append(r_VRP[math.floor((i-1)/2)]) #insert VRP key point
                self._addTrajRK45(duration, COM_func)
                self.timeSqu.append(self.timeSqu[-1] + sampTime) #update time sequences

            else: #continue to calculate COM until it coincides with the last DCM point
                VRP_Traj.append(r_VRP[-1]) #insert VRP key point
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
            plt.figure('double support trajectory',clear=True)
            fig = plt.figure('double support trajectory',figsize=(100,100))
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=60, azim=-60) #adjust view
            plt.cla()
            ax.plot([p[0] for p in self.DCM_Traj],
                    [p[1] for p in self.DCM_Traj],
                    [p[2] for p in self.DCM_Traj], color='blue', linewidth=1, label="DCM") #plot DCM trajectory
            ax.plot([p[0] for p in self.COM_Traj],
                    [p[1] for p in self.COM_Traj],
                    [p[2] for p in self.COM_Traj], color='green', linewidth=1, label="COM") #plot COM trajectory
            ax.plot([p[0] for p in VRP_Traj],
                    [p[1] for p in VRP_Traj],
                    [p[2] for p in VRP_Traj], color='red', linewidth=1, label="VRP") #plot VRP trajectory
            plt.legend(loc='upper right')

            ax.scatter([p[0] for p in r_VRP],
                        [p[1] for p in r_VRP],
                        [p[2] for p in r_VRP], color='black') #VRP points
            ax.scatter([p[0] for p in footPs],
                        [p[1] for p in footPs],
                        [p[2] for p in footPs], color='red') #foot points
            ax.scatter([p[0] for p in DCM_keySS],
                        [p[1] for p in DCM_keySS],
                        [p[2] for p in DCM_keySS], color='blue') #DCM SS key points
            ax.scatter([p[0] for p in DCM_key],
                        [p[1] for p in DCM_key],
                        [p[2] for p in DCM_key], color='yellow') #DCM key points(init and end points of double support)

            ax.set_xlabel('X(m)'); ax.set_ylabel('Y(m)'); ax.set_zlabel('Z(m)') #axis label

            lenth=np.abs(footPs[-1][1]-footPs[0][1]); 
            plt.xlim([-lenth/2,lenth/2]); plt.ylim([footPs[0][1],footPs[-1][1]]) #axis scale

            ax.set_title('Robot Trajetory(DCM,COM,VRP)(Double Support)') #set title

            text = '$t_{step,i}=%.2f(s)$\n$t_{DS,i}=%.2f(s)$\n$\\alpha_{DS}=%.2f$\ntotal time=%.2f(s)' % (timePS_SS, timePS_DS, DS_ratio, self.timeSqu[-1])
            ax.text2D(0.8, 0, text, color='red', transform=ax.transAxes) #text: total time

            #annote initial COM point and final COM point
            ax.scatter(initCOM[0],initCOM[1],initCOM[2], color='green')
            x0, y0, _ = proj3d.proj_transform(initCOM[0],initCOM[1],initCOM[2], ax.get_proj())
            plt.annotate(
                    r"$\mathbf{x}_{ini}$", xy = (x0, y0), xytext = (10, -50),
                    textcoords = 'offset points', ha = 'left', va = 'top',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
            
            finalCOM = self.COM_Traj[-1]
            ax.scatter(finalCOM[0],finalCOM[1],finalCOM[2], color='green')
            xf, yf, _ = proj3d.proj_transform(finalCOM[0],finalCOM[1],finalCOM[2], ax.get_proj())
            plt.annotate(
                    r"$\mathbf{x}_{end}$", xy = (xf, yf), xytext = (20, 50),
                    textcoords = 'offset points', ha = 'left', va = 'top',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))

            #annotate foot points and VRP, plot lines connecting VRP and foot points
            for i in range(len(footPs)): 
                f_point = footPs[i]; vrp_point = r_VRP[i]
                label1 = "$\\mathbf{r}_{f,%d}$" % (i+1)
                label2 = "$\\mathbf{r}_{vrp,%d}$" % (i+1)
                x1, y1, _ = proj3d.proj_transform(f_point[0],f_point[1],f_point[2], ax.get_proj())
                x2, y2, _ = proj3d.proj_transform(vrp_point[0],vrp_point[1],vrp_point[2], ax.get_proj())
                textpos = (-40, 40) if not(i%2) else (40, -40)
                horiz = 'left' if not(i%2) else 'right'; verti = 'top' if not(i%2) else 'bottom'
                plt.annotate(
                    label1, xy = (x1, y1), xytext = textpos,
                    textcoords = 'offset points', ha = horiz, va = verti,
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
                plt.annotate(
                    label2, xy = (x2, y2), xytext = textpos,
                    textcoords = 'offset points', ha = horiz, va = verti,
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
                ax.plot([f_point[0],vrp_point[0]], [f_point[1],vrp_point[1]], 
                        [f_point[2],vrp_point[2]], color='black', linewidth=1)

            #annotate DCM key points(init and end points of double support)
            for i in range(len(DCM_key)): 
                point = DCM_key[i]; 
                label = ("$\\mathbf{\\xi}_{iniDS,%d}$" % (math.floor(i/2)+1)) if not(i%2) else ("$\\mathbf{\\xi}_{eoDS,%d}$" % (math.floor(i/2)+1))
                textpos = (10, -30) if not(i%2) else (-10, 30)
                horiz = 'left' if not(i%2) else 'right'; verti = 'top' if not(i%2) else 'bottom'
                x1, y1, _ = proj3d.proj_transform(point[0],point[1],point[2], ax.get_proj())
                plt.annotate(
                label, xy = (x1, y1), xytext = textpos,
                textcoords = 'offset points', ha = horiz, va = verti,
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
            
            if savefig:  
                plt.savefig('DCM_DS.eps',format='eps')
                plt.savefig('DCM_DS.jpg',format='jpg')
            plt.show()

    #DCM planner for heel-toe double support motion(only in horizontal plain ground)
    def DCM_PlannerHT(self, initCOM, footPs, footRot, timePS_SS, timePS_DS, HT_ratio=0.5, DS_ratio=0.5, sampTime=0.01, plot=True, savefig=True) -> None:        
        #Step up basic parameters
        self.DCM_Traj=[] #clear existing DCM trajectory
        self.COM_Traj=[] #clear existing COM trajectory
        self.timeSqu = [] #clear existing time sequence
        g=9.8

        self.footPs = footPs
        HeelToePs = [] #heel and toe points
        for i in range(len(footPs)):
            d_len = self.footSize[0]/2 
            HeelToePs.append([footPs[i][0]+d_len*np.sin(footRot[i]), footPs[i][1]-d_len*np.cos(footRot[i]), footPs[i][2]]) #heel point
            HeelToePs.append([footPs[i][0]-d_len*np.sin(footRot[i]), footPs[i][1]+d_len*np.cos(footRot[i]), footPs[i][2]]) #toe point

        self.delz_vrp = initCOM[-1] - footPs[-1][-1] #walking stops at the last step(COM,VRP and DCM coincide)
        a = math.sqrt(g/self.delz_vrp) #A constant for DCM calculation
        self.timePS_SS = timePS_SS
        self.timePS_DS = timePS_DS 
        self.sampTime = sampTime
        t_HT = HT_ratio * timePS_SS #heel to toe time
        t_TH = timePS_SS - t_HT #toe to heel time
        t_iniDS = DS_ratio*timePS_DS #initial double support time(from the DCM key points in single support)
        t_endDS = timePS_DS - t_iniDS #end double support time(from the DCM key points in single support)

        #offset from foot points to construct VRPs
        r_VRP = list(map(lambda x: np.array(x)+[0,0,self.delz_vrp], HeelToePs))

        #Backward recursion
        DCM_keyHT = [] #store the DCM Heel & Toe key points
        dDCM_key = [] #DCM key points' derivatives
        DCM_iniDS = [] #DCM initial double support points
        dDCM_iniDS = [] #1-order derivative of DCM initial double support points
        DCM_endDS = [] #DCM end double support points
        dDCM_endDS = [] #1-order derivative of DCM end double support points
        for i in range(len(r_VRP)):
            newDCM_key = None
            if i==0: 
                DCM_keyHT.append(r_VRP[-1]) #recursion start from the end toe DCM key
            else:
                if (i%2): #Heel DCM key point
                    newDCM_key = r_VRP[-1-i] + math.exp(-a*t_HT)*(DCM_keyHT[-1]-r_VRP[-1-i]) #calculate new heel DCM key
                    DCM_keyHT.append(newDCM_key) #add new DCM key point to sequence

                    newDCM_endDS = r_VRP[-1-i] + math.exp(a*t_endDS)*(newDCM_key-r_VRP[-1-i]) #calculate nearby end DS key
                    DCM_endDS.append(newDCM_endDS) #add new end DS key point to sequence
                    newdDCM_endDS = a*math.exp(a*t_endDS)*(newDCM_key-r_VRP[-1-i]) #calculate the derivative of nearby end DS key
                    dDCM_endDS.append(newdDCM_endDS) #add it to sequence
                    
                    if i<(len(r_VRP)-1): #initial DS key point in normal condition
                        newDCM_iniDS = r_VRP[-2-i] + math.exp(-a*t_iniDS)*(newDCM_key-r_VRP[-2-i]) #calculate nearby initial DS key
                        DCM_iniDS.append(newDCM_iniDS) #add new initial DS key point to sequence
                        newdDCM_iniDS = a*math.exp(-a*t_iniDS)*(newDCM_key-r_VRP[-2-i]) #calculate nearby initial DS key
                        dDCM_iniDS.append(newdDCM_iniDS) 
                    else: #the first initial DS key point
                        DCM_iniDS.append(newDCM_key) #the first initial DS key point is the first DCM key
                        newdDCM_iniDS = a*(newDCM_key - r_VRP[-1-i])
                        dDCM_iniDS.append(newdDCM_iniDS) 

                else: #toe DCM key point
                    newDCM_key = r_VRP[-1-i] + math.exp(-a*t_TH)*(DCM_keyHT[-1]-r_VRP[-1-i]) #calculate new toe DCM key
                    DCM_keyHT.append(newDCM_key) #add new DCM key point to sequence

        DCM_key = [] #DCM key points(double support)
        DCM_keyHT.reverse() #reverse the heel and toe DCM SS initial points sequence to be forward
        DCM_iniDS.reverse() #reverse the backward DCM DS initial points sequence to be forward
        DCM_endDS.reverse() #reverse the backward DCM DS end points sequence to be forward
        dDCM_iniDS.reverse() #reverse the backward DCM DS initial points' derivatives sequence to be forward
        dDCM_endDS.reverse() #reverse the backward DCM DS end points' derivatives sequence to be forward

        #Rearrange DCM DS initial points and end points into one squence
        for i in range(len(DCM_iniDS)):
            DCM_key.append(DCM_iniDS[i]); DCM_key.append(DCM_endDS[i])
            dDCM_key.append(dDCM_iniDS[i]); dDCM_key.append(dDCM_endDS[i])

        DCM_key.append(DCM_keyHT[-1]); dDCM_key.append(np.array([0,0,0]))
        del DCM_iniDS, DCM_endDS, dDCM_iniDS, dDCM_endDS #delete unuseful variables

        #Forward step sampling, generate DCM trajetory and COM trajectory
        VRP_Traj = [] #VRP trajectory
        self.timeSqu.append(0) #start time sequence
        self.COM_Traj.append(np.array(initCOM)) #insert the first COM point
        for i in range(len(DCM_key)):
            DCM_equ=[]; COM_func=[]; dDCM_equ=[]; duration=None #reset variables
            self.DCM_Traj.append(DCM_key[i]) #insert DCM key point 
            if i<=1:
                VRP_Traj.append(r_VRP[0]) #insert VRP key point
            else:
                VRP_Traj.append(r_VRP[i-1]) #insert VRP key point

            if i < len(DCM_key)-1:  #calculate DCM and COM(RK45 solver) until DCM reach the end  
                if not(i%2): #index of initDS points
                    if i==0: #first initDS point
                        duration = t_endDS
                    else:
                        duration = timePS_DS
                else: #index of endDS points
                    duration = (timePS_SS - timePS_DS) if i<(len(DCM_key)-2) else (t_HT-t_endDS)

                DCM_equ, dDCM_equ, COM_func = self._COM_DCM_equDS(DCM_key[i], DCM_key[i+1], dDCM_key[i], 
                                                                dDCM_key[i+1], a, duration) #reconstruct lambda function

                for k in range(1,int(duration/sampTime)):
                    t_dcm = k*sampTime
                    self.timeSqu.append(self.timeSqu[-1] + sampTime) #update time sequences

                    self.DCM_Traj.append(DCM_equ(t_dcm))
                    VRP_Traj.append(DCM_equ(t_dcm) - (1/a)*dDCM_equ(t_dcm)) #insert VRP key point
                    
                self._addTrajRK45(duration, COM_func)
                self.timeSqu.append(self.timeSqu[-1] + sampTime) #update time sequences

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
            plt.figure('Heel to toe trajectory',clear=True)
            fig = plt.figure('Heel to toe trajectory',figsize=(100,100))
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=60, azim=-60) #adjust view
            plt.cla()
            ax.plot([p[0] for p in self.DCM_Traj],
                    [p[1] for p in self.DCM_Traj],
                    [p[2] for p in self.DCM_Traj], color='blue', linewidth=1, label="DCM") #plot DCM trajectory
            ax.plot([p[0] for p in self.COM_Traj],
                    [p[1] for p in self.COM_Traj],
                    [p[2] for p in self.COM_Traj], color='green', linewidth=1, label="COM") #plot COM trajectory
            ax.plot([p[0] for p in VRP_Traj],
                    [p[1] for p in VRP_Traj],
                    [p[2] for p in VRP_Traj], color='red', linewidth=1, label="VRP") #plot VRP trajectory
            plt.legend(loc='upper right')

            ax.scatter([p[0] for p in r_VRP],
                        [p[1] for p in r_VRP],
                        [p[2] for p in r_VRP], color='black') #VRP points
            ax.scatter([p[0] for p in HeelToePs],
                        [p[1] for p in HeelToePs],
                        [p[2] for p in HeelToePs], color='red') #foot points
            ax.scatter([p[0] for p in DCM_keyHT],
                        [p[1] for p in DCM_keyHT],
                        [p[2] for p in DCM_keyHT], color='blue') #DCM SS H-T or T-H key points
            ax.scatter([p[0] for p in DCM_key[0:-1]],
                        [p[1] for p in DCM_key[0:-1]],
                        [p[2] for p in DCM_key[0:-1]], color='yellow') #DCM key points(init and end points of double support)

            ax.set_xlabel('X(m)'); ax.set_ylabel('Y(m)'); ax.set_zlabel('Z(m)') #axis label

            lenth=np.abs(HeelToePs[-1][1]-HeelToePs[0][1])+0.5*2; 
            plt.xlim([-lenth/2,lenth/2]); plt.ylim([HeelToePs[0][1]-0.5,HeelToePs[-1][1]+0.5]) #axis scale

            ax.set_title('Robot Trajetory(DCM,COM,VRP)(Heel to Toe)') #set title

            text = '$t_{step,i}=%.2f(s)$\n$t_{DS,i}=%.2f(s)$\n$\\alpha_{DS}=%.2f$\n$\\alpha_{HT}=%.2f$\ntotal time=%.2f(s)' \
                                                                        % (timePS_SS, timePS_DS, DS_ratio, HT_ratio, self.timeSqu[-1])
            ax.text2D(0.8, 0, text, color='red', transform=ax.transAxes) #text: total time

            #annote initial COM point and final COM point
            ax.scatter(initCOM[0],initCOM[1],initCOM[2], color='green')
            x0, y0, _ = proj3d.proj_transform(initCOM[0],initCOM[1],initCOM[2], ax.get_proj())
            plt.annotate(
                    r"$\mathbf{x}_{ini}$", xy = (x0, y0), xytext = (10, -50),
                    textcoords = 'offset points', ha = 'left', va = 'top',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
            
            finalCOM = self.COM_Traj[-1]
            ax.scatter(finalCOM[0],finalCOM[1],finalCOM[2], color='green')
            xf, yf, _ = proj3d.proj_transform(finalCOM[0],finalCOM[1],finalCOM[2], ax.get_proj())
            plt.annotate(
                    r"$\mathbf{x}_{end}$", xy = (xf, yf), xytext = (20, 50),
                    textcoords = 'offset points', ha = 'left', va = 'top',
                    arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))

            #annotate VRP, plot lines connecting VRP and foot points
            for i in range(len(HeelToePs)): 
                f_point = HeelToePs[i]; vrp_point = r_VRP[i]
                label1 = "$\\mathbf{r}_{vrp,%d}$" % (i+1)
                x1, y1, _ = proj3d.proj_transform(vrp_point[0],vrp_point[1],vrp_point[2], ax.get_proj())
                textpos = (-40, 40) if not((i+1)%3) else (40, -40)
                horiz = 'left' if not((i+1)%3) else 'right'; verti = 'top' if not((i+1)%3) else 'bottom'
                # plt.annotate(
                #     label1, xy = (x1, y1), xytext = textpos,
                #     textcoords = 'offset points', ha = horiz, va = verti,
                #     arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
                ax.plot([f_point[0],vrp_point[0]], [f_point[1],vrp_point[1]], 
                        [f_point[2],vrp_point[2]], color='black', linewidth=1)

            #annotate DCM key points(init and end points of double support)
            for i in range(len(DCM_key)-1): 
                point = DCM_key[i]; 
                label = ("$\\mathbf{\\xi}_{iniDS,%d}$" % (math.floor(i/2)+1)) if not(i%2) else ("$\\mathbf{\\xi}_{eoDS,%d}$" % (math.floor(i/2)+1))
                textpos = (10, -30) if not(i%2) else (-10, 30)
                horiz = 'left' if not(i%2) else 'right'; verti = 'top' if not(i%2) else 'bottom'
                x1, y1, _ = proj3d.proj_transform(point[0],point[1],point[2], ax.get_proj())
                plt.annotate(
                label, xy = (x1, y1), xytext = textpos,
                textcoords = 'offset points', ha = horiz, va = verti,
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3'))
            
            if savefig:  
                plt.savefig('DCM_HT.eps',format='eps')
                plt.savefig('DCM_HT.jpg',format='jpg')
            plt.show()

if __name__=="__main__":
    robot = bipalRobot(80,[0.3,0.1])
    footSteps = [[-0.3, 0.0, 0.0],
                [0.3, 0.8, 0.0],
                [-0.3, 0.8*2, 0.0],
                [0.3, 0.8*3, 0.0],
                [-0.3, 0.8*4, 0.0],
                [0.3, 0.8*5, 0.0]]
    initialCOM = [0.0, 0.0, 1]

    #robot.DCM_PlannerSS(initialCOM, footSteps, 0.6) #single support test
    #robot.DCM_PlannerDS(initialCOM, footSteps, 0.6, 0.3, DS_ratio=0.4) #double support test

    footRot = [0]*6
    robot.DCM_PlannerHT(initialCOM, footSteps, footRot, 0.6, 0.2, HT_ratio=0.5, DS_ratio=0.6) #heel to toe test
