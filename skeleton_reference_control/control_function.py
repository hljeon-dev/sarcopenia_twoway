from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import cv2
from ultralytics.utils.plotting import Colors

def change_cal(target,theta,base):
    x = (target[0]-base[0])*math.cos(theta*math.pi/180) - (target[1] - base[1]) * math.sin(theta*math.pi/180) + base[0]
    y = (target[0]-base[0])*math.sin(theta*math.pi/180) - (target[1] - base[1]) * math.cos(theta*math.pi/180) + base[1]
    return np.array([int(x),int(y)])

def equation_of_a_line(pt,base):
    a= (pt[1]-base[1])/(pt[0]-base[0])
    b= pt[1] - (a*pt[0])
    return a,b

def equation_of_triangle_composition(center,base,radius):
    arl = base[0] - center [0]
    beta = base[1] - center[1]
    x = arl / math.sqrt(arl**2 + beta**2) * radius
    y = beta / math.sqrt(arl**2 + beta**2) *radius
    x += center [0]
    y += center [1]
    try:
        return  [int(x),int(y)]
    except: return [base[0],base[1]]

def estimate_pose_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

def equation_of_radiaus(pt1,pt2):
    return int(np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)) 

def affine(coords, m= -0.6):
    body_y = [coords[2][1],coords[3][1],coords[8][1],coords[9][1]]
    
    ori_body_y = min(body_y)+((max(body_y) - max(body_y))//2)
    result_x = []
    result_y = []
    cha_body= 0
    for coord in coords:
        x,y =coord
        y = m*x +y
        result_x.append(int(x))
        result_y.append(int(y))
    
    aff_body = [result_y[2], result_y[3], result_y[8], result_y[9]]
    
    aff_body_y = min(aff_body)+((max(aff_body) - max(aff_body))//2)

    cha_body = ori_body_y - aff_body_y
    result_y_a = result_y.copy()
    result_y = np.array(result_y) + cha_body
    
    result=[]
    for x,y in zip(result_x,result_y):
        result.append([x,y])

    return np.array(result)

def set_minmax_range(elbow_min, elbow_max, shoulder_min, shoulder_max, elbow_gradient, shoulder_gradient,
                     knee_min, knee_max, hip_min, hip_max, knee_gradient, hip_gradient, out_path, number, speed, important_angle):
    '''
    out_path는 동영상 저장 경로, number는 횟수 
    important_angle = ["elbow", "shoulder", "knee", "hip"]
    upper는 상체 운동, lower는 하체 운동
    '''

    colors = Colors()

    kpt_color = colors.pose_palette[[17, 17, 17, 18, 18, 18, 17, 17, 17, 18, 18, 18]]
    limb_color = colors.pose_palette[[ 17, 17, 18, 18, 16, 17, 18, 16,17, 17, 18, 18,]]

    skeleton = [
                [1,2],
                [2,3],
                
                [4,5],
                [5,6],
                
                [3,4],
                [3,9],
                [4,10],
                [9,10],

                [9,8],
                [8,7],
                [10,11],
                [12,11],
                
            ]

    kpts = [777, 607],[798, 480],[828, 341],[977, 339],[1010, 475], [1027, 593], [870, 817], [852, 709],[858, 554],	[954, 553],	[963, 715],	[973, 819]
    kpts = np.array(kpts)
    w,h = 1920, 1080
    radius=5
    kpt_line=True
    shape = (h,w)
    is_pose = True

    elbow_min_range, elbow_max_range = elbow_min, elbow_max 
    shoulder_min_range, shoulder_max_range = shoulder_min, shoulder_max
    elbow_gradient = elbow_gradient
    knee_min_range, knee_max_range = knee_min, knee_max 
    hip_min_range, hip_max_range = hip_min, hip_max 
    knee_gradient = knee_gradient

    pt0_theta_range = [elbow_min_range,elbow_max_range]
    pt0_theta_taget_index = 1
    pt0_theta = pt0_theta_range[0]

    pt1_theta_range = [shoulder_min_range,shoulder_max_range]
    pt1_theta_taget_index = 1
    pt1_theta = pt1_theta_range[0]

    pt4_theta_range = [-shoulder_max_range,-shoulder_min_range]
    pt4_theta_taget_index = 0
    pt4_theta = pt4_theta_range[1]

    pt5_theta_range = [elbow_min_range,elbow_max_range]
    pt5_theta_taget_index = 1
    pt5_theta = pt5_theta_range[0]

    pt6_theta_range = [knee_min_range,knee_max_range]
    pt6_theta_taget_index = 1
    pt6_theta = pt6_theta_range[0]

    pt7_theta_range = [hip_min_range,hip_max_range]
    pt7_theta_taget_index = 1
    pt7_theta = pt7_theta_range[0]

    pt10_theta_range = [-hip_max_range,-hip_min_range]
    pt10_theta_taget_index = 0
    pt10_theta = pt10_theta_range[1]

    pt11_theta_range = [knee_min_range,knee_max_range]
    pt11_theta_taget_index = 1
    pt11_theta = pt11_theta_range[0]

    pt0_radius = equation_of_radiaus(kpts[0],kpts[1])
    pt1_radius = equation_of_radiaus(kpts[1],kpts[2])

    pt4_radius = equation_of_radiaus(kpts[3],kpts[4])
    pt5_radius = equation_of_radiaus(kpts[4],kpts[5])

    pt6_radius = equation_of_radiaus(kpts[6],kpts[7])
    pt7_radius = equation_of_radiaus(kpts[7],kpts[8])

    pt10_radius = equation_of_radiaus(kpts[9],kpts[10])
    pt11_radius = equation_of_radiaus(kpts[10],kpts[11])

    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D','I','V','X'
    path = out_path
    out = cv2.VideoWriter(path, fourcc, 30, (w, h))

    h_direction = True
    kn_direction = True
    num = 0

    if len(important_angle) >= 2:
        number *= 2

    while num < number:
        kpts = [777, 607],[798, 480],[828, 341],[977, 339],[1010, 475], [1027, 593], [870, 817], [852, 709],[858, 554],	[954, 553],	[963, 715],	[973, 819]
        kpts = np.array(kpts)

        pre_skeleton = np.full((h,w,3), (255, 255, 255), dtype=np.uint8)
        
    #오른팔 
        #오른팔꿈치
        pt1_target = [kpts[2][0] + pt1_radius,kpts[2][1]]
        pt1_target =change_cal(pt1_target,shoulder_gradient,kpts[2])
        pt1 = change_cal(pt1_target,pt1_theta,kpts[2])

        if "shoulder" in important_angle:
            if pt1_theta_taget_index ==1 and pt1_theta < pt1_theta_range[pt1_theta_taget_index]:
                pt1_theta +=speed
            elif pt1_theta_taget_index ==1 and pt1_theta == pt1_theta_range[pt1_theta_taget_index]:
                pt1_theta_taget_index = 0
            elif pt1_theta_taget_index == 0 and pt1_theta > pt1_theta_range[pt1_theta_taget_index]:
                pt1_theta -=speed
            elif pt1_theta_taget_index == 0 and pt1_theta == pt1_theta_range[pt1_theta_taget_index]:
                pt1_theta_taget_index = 1    
        
        #오른손목
        pt0_target =[pt1[0]+pt0_radius,pt1[1]]
        pt0_target =change_cal(pt0_target,elbow_gradient,pt1)
        pt0 = change_cal(pt0_target,pt1_theta + pt0_theta,pt1)
        
        if "elbow" in important_angle:
            if pt0_theta_taget_index ==1 and pt0_theta < pt0_theta_range[pt0_theta_taget_index]:
                pt0_theta +=speed
            elif pt0_theta_taget_index ==1 and pt0_theta == pt0_theta_range[pt0_theta_taget_index]:
                pt0_theta_taget_index = 0
            elif pt0_theta_taget_index == 0 and pt0_theta > pt0_theta_range[pt0_theta_taget_index]:
                pt0_theta -=speed
            elif pt0_theta_taget_index == 0 and pt0_theta == pt0_theta_range[pt0_theta_taget_index]:
                pt0_theta_taget_index = 1
            

    #왼팔   
        #왼팔꿈치   
        pt4_target = [kpts[3][0] - pt4_radius,kpts[3][1]]
        pt4_target =change_cal(pt4_target,-shoulder_gradient,kpts[3])
        pt4 = change_cal(pt4_target,pt4_theta,kpts[3])

        if "shoulder" in important_angle:
            if pt4_theta_taget_index ==1 and pt4_theta < pt4_theta_range[pt4_theta_taget_index]:
                pt4_theta +=speed
            elif pt4_theta_taget_index ==1 and pt4_theta == pt4_theta_range[pt4_theta_taget_index]:
                pt4_theta_taget_index = 0
                num += 1
            elif pt4_theta_taget_index == 0 and pt4_theta > pt4_theta_range[pt4_theta_taget_index]:
                pt4_theta -=speed 
            elif pt4_theta_taget_index == 0 and pt4_theta == pt4_theta_range[pt4_theta_taget_index]:
                pt4_theta_taget_index = 1    
        
        #왼손목  
        pt5_target =[pt4[0]-pt5_radius,pt4[1]]
        pt5_target =change_cal(pt5_target,-elbow_gradient,pt4)  
        pt5 = change_cal(pt5_target, pt4_theta-pt5_theta,pt4)
        
        if "elbow" in important_angle:
            if pt5_theta_taget_index ==1 and pt5_theta < pt5_theta_range[pt5_theta_taget_index]:
                pt5_theta +=speed
            elif pt5_theta_taget_index ==1 and pt5_theta == pt5_theta_range[pt5_theta_taget_index]:
                pt5_theta_taget_index = 0
                num += 1
            elif pt5_theta_taget_index == 0 and pt5_theta > pt5_theta_range[pt5_theta_taget_index]:
                pt5_theta -=speed
            elif pt5_theta_taget_index == 0 and pt5_theta == pt5_theta_range[pt5_theta_taget_index]:
                pt5_theta_taget_index = 1    

    #오른하체
        #오른힙
        pt7_target = [kpts[8][0] + pt7_radius,kpts[8][1]]
        pt7_target =change_cal(pt7_target,hip_gradient,kpts[8])
        pt7 = change_cal(pt7_target,pt7_theta,kpts[8])
        
        if "hip" in important_angle and h_direction == True:
            if pt7_theta_taget_index ==1 and pt7_theta < pt7_theta_range[pt7_theta_taget_index]:
                pt7_theta +=speed
            elif pt7_theta_taget_index ==1 and pt7_theta == pt7_theta_range[pt7_theta_taget_index]:
                pt7_theta_taget_index = 0
            elif pt7_theta_taget_index == 0 and pt7_theta > pt7_theta_range[pt7_theta_taget_index]:
                pt7_theta -=speed
            elif pt7_theta_taget_index == 0 and pt7_theta == pt7_theta_range[pt7_theta_taget_index]:
                pt7_theta_taget_index = 1 
                h_direction = False

        #오른무릎
        pt6_target =[pt7[0]+pt6_radius,pt7[1]]
        pt6_target =change_cal(pt6_target,knee_gradient,pt7)
        pt6 = change_cal(pt6_target,pt7_theta + pt6_theta,pt7)
        
        if "knee" in important_angle and kn_direction == True:
            if pt6_theta_taget_index ==1 and pt6_theta < pt6_theta_range[pt6_theta_taget_index]:
                pt6_theta +=speed
            elif pt6_theta_taget_index ==1 and pt6_theta == pt6_theta_range[pt6_theta_taget_index]:
                pt6_theta_taget_index = 0
            elif pt6_theta_taget_index == 0 and pt6_theta > pt6_theta_range[pt6_theta_taget_index]:
                pt6_theta -=speed
            elif pt6_theta_taget_index == 0 and pt6_theta == pt6_theta_range[pt6_theta_taget_index]:
                pt6_theta_taget_index = 1
                kn_direction = False    

    #왼하체
        #왼힙
        pt10_target = [kpts[9][0] - pt10_radius,kpts[9][1]]
        pt10_target = change_cal(pt10_target,-hip_gradient,kpts[9])
        pt10 = change_cal(pt10_target,pt10_theta,kpts[9])

        if "hip" in important_angle and h_direction == False:
            if pt10_theta_taget_index == 0 and pt10_theta > pt10_theta_range[pt10_theta_taget_index]:
                pt10_theta -=speed   
            elif pt10_theta_taget_index == 0 and pt10_theta == pt10_theta_range[pt10_theta_taget_index]:
                pt10_theta_taget_index = 1  
            elif pt10_theta_taget_index ==1 and pt10_theta < pt10_theta_range[pt10_theta_taget_index]:
                pt10_theta +=speed
            elif pt10_theta_taget_index ==1 and pt10_theta == pt10_theta_range[pt10_theta_taget_index]:
                pt10_theta_taget_index = 0
                h_direction = True
                num += 0.5
                

        #왼무릎
        pt11_target = [pt10[0]-pt11_radius,pt10[1]]
        pt11_target = change_cal(pt11_target,-knee_gradient,pt10)  
        pt11 = change_cal(pt11_target, pt10_theta-pt11_theta,pt10)
        
        if "knee" in important_angle and kn_direction == False:
            if pt11_theta_taget_index ==1 and pt11_theta < pt11_theta_range[pt11_theta_taget_index]:
                pt11_theta +=speed
            elif pt11_theta_taget_index ==1 and pt11_theta == pt11_theta_range[pt11_theta_taget_index]:
                pt11_theta_taget_index = 0
            elif pt11_theta_taget_index == 0 and pt11_theta > pt11_theta_range[pt11_theta_taget_index]:
                pt11_theta -=speed
            elif pt11_theta_taget_index == 0 and pt11_theta == pt11_theta_range[pt11_theta_taget_index]:
                pt11_theta_taget_index = 1
                kn_direction = True
                num += 0.5
                
        #초기화
        kpts[0] = pt0
        kpts[1] = pt1
        kpts[4] = pt4
        kpts[5] = pt5
        kpts[6] = pt6
        kpts[7] = pt7
        kpts[10] = pt10
        kpts[11] = pt11
        # kpts = affine(kpts)

        for i, k in enumerate(kpts):
            color_k = [int(x) for x in kpt_color[i]] if is_pose else colors(i)
            
            x_coord, y_coord = k[0], k[1]
        
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
            
            cv2.circle(pre_skeleton, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue

            cv2.line(pre_skeleton, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        out.write(pre_skeleton)
        cv2.imshow('pre_sk',pre_skeleton)
        
        k = cv2.waitKey(1) & 0xFF
        if(k == 27):
            break

    cv2.destroyAllWindows()

