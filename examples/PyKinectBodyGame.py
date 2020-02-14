from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import utils_PyKinectV2 as utils
import open3d
import cv2
import ctypes
import _ctypes
import pygame
import csv
import sys
import pickle
import numpy as np
from numpy.linalg import norm

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]
                  


class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        self._key = None

        


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # # Right Leg
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # # Left Leg
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()
        

    def run(self):
        # -------- Main Program Loop -----------
        c1 = 400
        c2 = 400
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._done = True

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
            # --- Game logic should go here
            
            
            # --- Getting frames and drawing  
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F1:
                            c1 += 1 
                            utils.save_rgb(frame,"_frame"+str(c1))

                if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F2:
                            c2 += 1 
                            utils.save_rgb2(frame,"_frame"+str(c2))            
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None: 
                for i in range(0, self._kinect.max_body_count):
                # for i in range(0, 1):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints
                    body_frame  = self._kinect.get_last_body_frame()
                    depth_frame = self._kinect.get_last_depth_frame()

                    depth_width, depth_height = self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height # Default: 512, 424

                    depth_img   = depth_frame.reshape(((depth_height, depth_width))).astype(np.uint16)

                    ppx=260.166; ppy=205.197; fx=367.535; fy=367.535; depth_scale=0.001;
                    intrinsic = open3d.camera.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, ppx, ppy) 

                    
                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    
                    #use the utils.py file to get the 3D joints in meters
                    joint3D, orientation = utils.get_single_joint3D_and_orientation(self._kinect, body_frame, depth_img, intrinsic, depth_scale)   

                    norma_1, theta_1, phi_1 = utils.get_vector_info(joint3D[8], joint3D[9])  #trace vector 1 (shoulder to elbow)
                    norma_2, theta_2, phi_2 = utils.get_vector_info(joint3D[9], joint3D[10])  #trace vector 2 (elbow to wrist)
                    norma_3, theta_3, phi_3 = utils.get_vector_info(joint3D[10], joint3D[11])  #trace vector 3 (wrist to hand)

                    print("Sample: " + str(joint3D[11]))
                    # print("V2: " + str(theta_2))
                    # print("V3: " + str(theta_3))

                    # # # print if the right hand is over the right marker in the space
                    # if joint3D[11,0] > 0.4 and joint3D[11,0] < 0.45 and joint3D[11,1] > 0.14 and joint3D[11,1] < 0.15 and joint3D[11,2] > 0.84 and joint3D[11,2] < 0.88:
                    # # if joint3D[11,0] > -0.01 and joint3D[11,0] < 0.1 and joint3D[11,1] > 0.15 and joint3D[11,1] < 0.17 and joint3D[11,2] > 0.79 and joint3D[11,2] < 0.85:
                    #     print("Right hand INSIDE the area")
                    #     self._done = True

                    # else:
                    #     print("Right hand OUTSIDE the area")
                    
                    # # let's decide what's the state of the right hand of the body
                    
                    # if (body.hand_right_state == 2 and body.hand_right_confidence == 1):
                    #     print('Right hand is OPEN')

                    # if (body.hand_right_state == 3 and body.hand_right_confidence == 1):
                    #     print('Right hand is CLOSED')

                    

                    # save csv file with the data frames aquired with F1 key
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F1:
                            print("save data")
                            utils.save_csv(joint3D[8,0], joint3D[8,1], joint3D[8,2],
                            joint3D[9,0], joint3D[9,1], joint3D[9,2],
                            joint3D[10,0], joint3D[10,1], joint3D[10,2], 
                            joint3D[11,0], joint3D[11,1], joint3D[11,2],
                            theta_1,phi_1,
                            theta_2,phi_2,
                            theta_3,phi_3,norma_3,
                            c1)

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_F2:
                            print("save data")
                            utils.save_csv2(joint3D[8,0], joint3D[8,1], joint3D[8,2],
                            joint3D[9,0], joint3D[9,1], joint3D[9,2],
                            joint3D[10,0], joint3D[10,1], joint3D[10,2], 
                            joint3D[11,0], joint3D[11,1], joint3D[11,2],
                            theta_1,phi_1,
                            theta_2,phi_2,
                            theta_3,phi_3,norma_3,
                            c2)        

                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])

            

            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(10)

        # Close our Kinect sensor, close the window and quit.
        
        self._kinect.close()
        pygame.quit()
        
        


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();

