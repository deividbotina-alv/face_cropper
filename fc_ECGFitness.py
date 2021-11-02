'''
This cript takes the ECG-Fitness dataset and cropp the faces taking the bounding boxes given by the authors, saving them with their
respective ground truth files.
Steps:
    
'''
#%% GLOBAL VARIABLES
loadingPath = r'J:\Original_Datasets\VIPL-HR\data' # Path where we can find original files
savingPath = r'J:\faces\128_128\original\VIPL' # Path where faces will be saved
filespath = r'E:\repos\face_cropper\source' #Path with the files needed to run this script
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,plot,show,legend,pause,title
import os
import subprocess
from os import listdir
from os.path import join
import shutil as sh
from natsort import natsorted
import glob
import sys
import cv2
import copy

#%% CLASSES AND FUNCTIONS

def get_number_of_files(path,typefile=''):
    n_files = 0
    for root, dirs, files in os.walk(path):
        for i in files:
            if i.endswith(typefile):
                n_files = n_files + 1

    return n_files
    

class face_cropper_VIPL():
    # CONSTRUCTOR
    def __init__(self,LoadPath,SavePath,filespath,SHOW=False):
        self.LoadPath = LoadPath # Path with the dataset
        self.SavePath = SavePath # Path where faces will be saved
        self.filespath = filespath # Path with the files needed to run this script
        self.SHOW = SHOW # Show FACES croping
        self.confidenceThreshold = 0.5 
        self.critical_confidence = 0.99#if confidence is lower than critical_confidence, save the frame in text file (i.e. bad frame)
    
    # FUNCTION TO FIND PATHS WHERE DATA AND GT FILES ARE LOCATED
    def find_files(self):
        # CREATE FOLDER WHERE RESULTS WILL BE SAVED
        if not os.path.exists(self.SavePath): os.makedirs(self.SavePath)
        # Finding Folders where ground truth located
        folders_gt = [] 
        for root, dirs, files in os.walk(self.LoadPath):
            for i in files:
                if i.endswith("wave.csv"):
                    if not(root.split(os.path.sep)[-1].lower()=='source4'):#All sources minus source4(NIR)
                        folders_gt.append(root)
                        break

        # Finding Folders where the data is located
        folders_images = [] 
        for root, dirs, files in os.walk(self.LoadPath):
            for i in files:
                if i.endswith("video.avi"):
                    if not(root.split(os.path.sep)[-1].lower()=='source4'):#All sources minus source4(NIR)
                        folders_images.append(root)
                        break
        
        #Natural sort in files
        self.datapath = np.array(natsorted(folders_images))
        self.gtpath =  np.array(natsorted(folders_gt))
        
    # FUNCTION TO SET PATHS WHIT SUBJECTS
    def set_files(self, folders_images, folders_gt):
        #Natural sort in files
        self.datapath = np.array(natsorted(folders_images))
        self.gtpath =  np.array(natsorted(folders_gt))

    # Flag Checking if there is the same number of data files that the ground truth files
    def same_number_of_files(self):        
        return np.size(self.datapath) == np.size(self.gtpath)
    
    # FUNCTION TO REPORT SOMETHING IN A .TXT FILE
    def report_TXT(self,nameFile,message):
        fileTXT = open(nameFile,"a")
        fileTXT.write(message)
        fileTXT.close()
        
    # FUNCTION TO COPY THE ORIGINAL GROUND TRUTH FILES AND PASTE THEM IN SAVE PATH
    def copy_GT_files(self):
        # Dataset report
        print('[CG]Copy and pasting ground truth files...')
        self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                '\nCOPY AND PASTE GROUND TRUTH FILES\n\n')
        for path in self.gtpath:# Number of subjects 
            subject = path.split(os.path.sep)[-3]+path.split(os.path.sep)[-2]+path.split(os.path.sep)[-1][0]+path.split(os.path.sep)[-1][-1] #subject name
            # Create folder for current subject if it does not exist
            if not os.path.exists(join(self.SavePath,subject)): os.makedirs(join(self.SavePath,subject))
            
            # Copy the respective ground truth with the same name but in .txt file
            try:
                sh.copy(join(path,'wave.csv'),join(self.SavePath,subject,subject+'_gt.csv'))
                self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                    'Subject ' + subject + ': GT file OK\n')
            except:
                self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                    'Subject ' + subject + ': ERROR saving GT file\n')
                
            # Source 1 and 3 in VIPL have timestamp file. Copy and paste it.
            if subject[-1]=='1'or subject[-1]=='3':
                try:
                    sh.copy(join(path,'time.txt'),join(self.SavePath,subject,subject+'_timestamp.txt'))
                    self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        'Subject ' + subject + ': time stamp file OK\n')
                except:
                    self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        'Subject ' + subject + ': ERROR saving time stamp file\n')
            elif subject[-1]=='2':
                try:
                    #Create timestamp file for souce 2 30fps 
                    sr = 0.033
                    video = cv2.VideoCapture(join(path,'video.avi'))
                    totalframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    timeTrace = np.zeros(totalframes)
                    for j in range(0,len(timeTrace)):
                        timeTrace[j] = j*sr#sr=0.033
                    text_file = open(join(self.SavePath,subject,subject+'_timestamp.txt'), "w")
                    for val in timeTrace:                        
                        text_file.write(f'{int(round(val*1000))}\n')
                    text_file.close()
                    self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        'Subject ' + subject + ': time stamp manually created at 30 hz\n')
                except:
                    self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        'Subject ' + subject + ': ERROR saving time stamp file\n')
        print('[CG]Process completed')
    # FUNCTION TO DETECT THE FACE IN A SPECIFIC FRAME: Usually used in first frame or whenever the tracker lost the face.
    def detect_face(self, frame):
        
        face_found = False
        #1) TRY TO FIND FACE BY Caffe network
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,#image,scalefactor,size
                                     (300, 300), (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        best_detection = None
        maxConfidence = 0
        
        # IF THE NETWORK FOUND FACES
        if detections.shape[2]>0:

        # RETURN ONLY THE BEST DETECTION WITH THE MAXCONFIDENCE 
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the prediction
                confidence = detections[0, 0, i, 2]
        
                # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
                if confidence < self.confidenceThreshold or confidence < maxConfidence:
                    continue
                
                maxConfidence = confidence
                best_detection = detections[0, 0, i, 3:7]
                
                face_found = True
        # IF CAFFE COULDN'T FIND ANY FACE
        else:
            face_found = False
            
        # #2) USE FACECASCADE INSTEAD -- NOT FINISHED NOR TESTED YET
        # if not(face_found):
        #     # We have to send the best_detection as the coffenetwork would do it, so we divide facecascade results by H and W
        #     (H, W) = frame.shape[:2]
        #     face_box = self.faceCascade.detectMultiScale(
        #         frame,
        #         scaleFactor=1.1,
        #         minNeighbors=5,
        #         minSize=(30, 30),
        #         flags = cv2.CASCADE_SCALE_IMAGE
        #     )
        #     # Take only the first face founded
        #     if len(face_box) < 1:
        #         maxConfidence = 0
        #         best_detection = None
        #     else: # If there is one or more faces                
        #         face_box = np.resize(face_box[0],(1,4)).copy()
        #         (x, y, w, h) = (face_box[0,0],face_box[0,1],face_box[0,2],face_box[0,3])
    
        #         maxLength = max(h, w)
        #         x -= int((maxLength-w)/2)
        #         y -= int((maxLength-h)/2)
        #         h = maxLength
        #         w = maxLength

            
        return (maxConfidence, best_detection)

    # FUNCTION TO GET THE NUMBER OF FRAMES IN ONE VIDEO
    def get_number_frames(self,subject_path):
        video = cv2.VideoCapture(os.path.join(subject_path,'video.avi'))
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))#Opencv 3
        video.release()
        return total
        

    # FUNCTION TO EXTRACT THE FRAMES NAMES INSIDE THE CURRENT FOLDER
    def get_frames_names(self,subject_path): #counf frames in one subject folder
            n_framesL = []
            for file in os.listdir(subject_path):
                if file.endswith(".jpg"):
                    n_framesL.append(join(subject_path,file))
            n_framesL = np.array(natsorted(n_framesL))
            return n_framesL# List with the names per frame

    # FUNCTION TO RETURN CROP IMAGE ACCORDING TO THE BOS
    def crop_image(self,frame, box):
        (x, y, w, h) = [int(v) for v in box]
        
        maxLength = max(h, w)
        x -= int((maxLength-w)/2); 
        if x<0: x=0;
        y -= int((maxLength-h)/2);
        if y<0: y=0;
        h = maxLength
        w = maxLength

        crop_img = copy.deepcopy(frame[y:y+h, x:x+w])
        
        return crop_img

    def crop_faces(self,new_dimensions):
        
        print('[CF]Cropping faces by Network from caffe...')
        # LOAD CAFFE NETWORK FOR FACE DETECTION
        try:
            self.net = cv2.dnn.readNetFromCaffe(join(self.filespath,"deploy.prototxt.txt"),join(self.filespath,"res10_300x300_ssd_iter_140000.caffemodel"))
        except:
            print('[CF]ERROR: '+ join(self.filespath,'deploy.prototxt.txt or res10_300x300_ssd_iter_140000.caffemodel')+' not found')
            sys.exit()
        # LOAD FACECASCADE FOR FACE DETECTION
        try: 
            self.faceCascade = cv2.CascadeClassifier(os.path.join(self.filespath,'haarcascade_frontalface_default.xml'))
        except:
            print('[CF]ERROR: '+os.path.join(self.filespath,'haarcascade_frontalface_default.xml')+' not found')
            sys.exit()
        # Start report of current dataset
        self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                        "[INFO] Face cropping report of dataset in " + self.LoadPath + '\n')
        
        # Crop faces subject by subject
        for i in range(0,len(self.datapath)):# Number of subjects
            subject = self.datapath[i].split(os.path.sep)[-3]+self.datapath[i].split(os.path.sep)[-2]+self.datapath[i].split(os.path.sep)[-1][0]+self.datapath[i].split(os.path.sep)[-1][-1]#subject name
            print('[CF]Subject: '+ subject)
            # Get names of all frames in current subject
            n_framesL = self.get_number_frames(self.datapath[i])
            # Check if current subject was saved already (it must have all frames saved)
            n_files = get_number_of_files(os.path.join(self.SavePath,subject),typefile='.png')#
            
            if n_files>=n_framesL:#If this subject is done
                pass
            else:
                unusual_subject = False #Flag refering to wether all faces were found and cropped properly in all frames
                count_unusual = 0
                has_to_detect_face = True #Flag refering if is first frame or the tracker lost the face
                initBB = None # initialize the bounding box coordinates of the object we are going to track
    
                # Create folder for current subject
                if not os.path.exists(join(self.SavePath,subject)): os.makedirs(join(self.SavePath,subject))
                # Start report in current subject
                self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                                "[INFO] This file corresponds to " + subject +
                                ".\nIt save problematic frames. If it is empty, each frame should be good.\n\n")
                video = cv2.VideoCapture(os.path.join(self.datapath[i],'video.avi'))
                for j in range(0,n_framesL):#Number of frames in this subject
                    # If current subject 
                    is_crop_img = False # Flag refering to wheter the face was cropped succesfully from current frame
                    crop_img = np.zeros((0,0,0))
                    current_frame_jpg = str(j).zfill(len(str(n_framesL)))+'.jpg'
                    current_frame_png = str(j).zfill(len(str(n_framesL)))+'.png' # subject_frame.png

                    try: # Try to find the face in current frame, crop, resize and save it
                        _,frame = video.read()
                        #Get size of frame
                        (H, W) = frame.shape[:2]
                        # 1) IF WE DON'T KNOW WHERE THE FACE IS: -> detect face
                        if has_to_detect_face:
                            print("[INFO] Launch face detection")
                            maxConfidence, best_detection = self.detect_face(frame)
    
                            if maxConfidence > 0:
                                # compute the (x, y)-coordinates of the bounding box for the object
                                box = best_detection * np.array([W, H, W, H])
    
                                (startX, startY, endX, endY) = box
                                # As we found the face, start tracking
                                initBB = (startX, startY, endX-startX, endY-startY)
                                tracker = cv2.TrackerMedianFlow_create()
                                tracker.init(frame, initBB)
                                # Crop the current face that we just found
                                crop_img = self.crop_image(frame, initBB)
                                is_crop_img = True
                                
                                (startX, startY, endX, endY) = box.astype("int")
                                
                                # draw the bounding box of the face along with the associated probability
                                text = "{:.2f}%".format(maxConfidence * 100)
                                y = startY - 10 if startY - 10 > 10 else startY + 10
                                cv2.rectangle(frame, (startX, startY), (endX, endY),
                                              (0, 0, 255), 2)
                                cv2.putText(frame, text, (startX, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                
                                has_to_detect_face = False
                                print("[INFO] Face found")

                        # 2) IF WE KNOW WHERE THE FACE IS, USE THE TRACKER
                        else: #hast_to_detect_face = False
                            (success, box) = tracker.update(frame)
                            
                            if success:# If it finds again the face
                                crop_img = self.crop_image(frame, box)
                                is_crop_img = True
                                
                                (crop_width, crop_height) = crop_img.shape[:2]
                                if crop_width > 0 and crop_height > 0:
                                    
                                    # If the face found by the tracker is not reliable, we should rerun facedetection in next frame
                                    maxConfidence, _ = self.detect_face(crop_img)                                    
                                    if maxConfidence < self.critical_confidence:
                                        has_to_detect_face = True
                                        
                                (x, y, w, h) = [int(v) for v in box]
                                cv2.rectangle(frame, (x, y), (x+w, y+h),
                                              (0, 255, 0), 2)
                            else: # if the tracker couldn't find the face
                                #Try to find the face without the tracker
                                print("[INFO] Launch face detection, tracker failed")
                                maxConfidence, best_detection = self.detect_face(frame)
                                if maxConfidence > 0:
                                    box = best_detection * np.array([W, H, W, H])    
                                    (startX, startY, endX, endY) = box
                                    initBB = (startX, startY, endX-startX, endY-startY)
                                    crop_img = self.crop_image(frame, initBB)
                                    is_crop_img = True
                                has_to_detect_face = True
                                
                        # SHOW FACE FOUND IN CURRENT FRAME IF NEEDED
                        if self.SHOW:
                            cv2.imshow("Frame", frame)
                        
                        # IF CROP FACE FOUND AND DIMMENSIONS ARE GOOD
                        if is_crop_img and crop_img.shape[0]>0 and crop_img.shape[1]>0:
                            if self.SHOW:
                                cv2.imshow("crop", crop_img)
                            # Resize face
                            crop_img = cv2.resize(crop_img, new_dimensions, interpolation = cv2.INTER_AREA)
                            cv2.imwrite(join(self.SavePath,subject,current_frame_png), crop_img)
                            if maxConfidence < self.critical_confidence:
                                self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                                                "[WARNING] Frame " + current_frame_jpg + " maxConfidence: "+ str(maxConfidence)+"\n")
                                unusual_subject = True; count_unusual += 1
                                print("[WARNING] Frame " + current_frame_jpg + " maxConfidence: "+ str(maxConfidence)+"\n")

                        else: # IF WE DIDN'T FIND A RELIABLE FACE, JUST SAVE AN EMPTY IMAGE TO PRESERVE THE SAME NUMBER OF FRAMES IN THE SUBJECT
                            empty = np.zeros(new_dimensions,dtype=float)
                            crop_img = cv2.merge((empty,empty,empty))
                            cv2.imwrite(join(self.SavePath,subject,current_frame_png), crop_img)
                            self.report_TXT(join(self.SavePath,subject,'Report.txt'),
                                "[ERROR] Frame " + current_frame_jpg + " Error in cropping, saving ZEROS as result\n")
                            unusual_subject = True; count_unusual += 1
                            print("[ERROR] Frame " + current_frame_jpg + " Error in cropping, saving ZEROS as result")
                    except: #Excpetion of: Try to find the face in current frame, crop, resize and save it
                        # report something wrong happened
                        self.report_TXT(join(self.SavePath,subject,'Report.txt'),'ERROR in Frame '+current_frame_jpg.split('.')[0]+'\n')
                        unusual_subject = True; count_unusual += 1
                        print('[ERROR]in Frame '+current_frame_jpg.split('.')[0])
                    # REPORT IF ALL FRAMES OF CURRENT SUBJECT WERE OK OR NOT
                    if j == n_framesL-1: # If last frame
                        if unusual_subject: #If either of the current subject's frames went wrong:
                             self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                                     "Subject " + subject + ': ' + str(count_unusual) + '/' + str(n_framesL)+'  Unusual frames\n')
                        else:# If all current subject's frames were fine:
                             self.report_TXT(join(self.SavePath,'Dataset_Report.txt'),
                                             
                                     "Subject " + subject + ': OK\n')
        print('[CF]Process completed')

def checkGroundTruthFilesInECGFitness(path:str):
    '''Plot all Ground truth signals to get the reliable segments for all subjects'''
    n_frames = 1800 # In
    # For all subjects
    time = np.arange(0,(1./30.)*n_frames,(1./30.))# viatom works at 125 Hz
    for sbj in Subjects:
        for (dirpath, dirnames, filenames) in os.walk(join(path,sbj)):
            for folder in dirnames:  
                GTfile = np.load(join(dirpath,folder,'fakePPG.npy'))
                figure(1),plt.cla(),title(sbj+'_'+folder),plot(time,GTfile),plt.grid(),show()
                print(sbj+'_'+folder)
  

#%% MAIN

loadingPath = r'J:\Original_Datasets\ECG-Fitness'
bboxPath = r'J:\Original_Datasets\ECG-Fitness\bbox'
Subjects = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']
savingPath = r'J:\faces\128_128\synchronized\ECGFitness'
SHOW_INDIVIDUAL_PREDICTION = True 
SHOW_FINAL_PREDICTION = True

checkGroundTruthFilesInECGFitness(loadingPath)
    
# CHECK GROUND TRUTH FILES
# For all subjects
for sbj in Subjects:
    for (dirpath, dirnames, filenames) in os.walk(join(loadingPath,sbj)):
        for folder in dirnames:
            
            try:
                file = np.load(join(dirpath,folder,'fakePPG.npy'))
                print('Previously saved: '+join(dirpath,folder,'fakePPG.npy'))
            except:   
                time,GT_ecg = getECGFromCSVFileECGFitness(join(dirpath,folder),SHOW=False)
                fakeppg = ECG2PPG_wgan_gp_mimic(ECG2PPGnet,time,GT_ecg,SHOW=False) 
                # save fakeppg                    
                np.save(join(dirpath,folder,'fakePPG.npy'),fakeppg.astype(np.float16))
                print(join(dirpath,folder,'fakePPG.npy'))


VIPL = face_cropper_VIPL(loadingPath, savingPath, filespath, SHOW=False)
VIPL.find_files()
# Test with one subject
# Uncomment next line to test one or multiple specific subjetcs
# VIPL.datapath = [r'J:\Original_Datasets\VIPL-HR\data\p1\v1\source2']
# VIPL.gtpath =  [r'J:\Original_Datasets\VIPL-HR\data\p1\v1\source2']
if not(VIPL.same_number_of_files()):
    print('Error: pathFiles and pathGT must have the same number of files')
    sys.exit()
else:
    VIPL.copy_GT_files()
    #VIPL.crop_faces((128,128))
