# FACE CROPPER AND SYNCHRONIZATION

## 1. FACE CROPPER

The first part of this repository is in the "source\fc_DATASET" scripts where DATASET can be either MMSE or VIPL (fc_MMSE.py and fc_VIPL.py). In this script the objective is to take the original videos or images from a database, crop only the face present in the scene and save it in .png format. This is done for all frames and all subjects.

![image](https://drive.google.com/uc?export=view&id=17uNN8JKcwkXprIv-05VftcROwmXgdCgP)

The paths where the original database is located, the output folder and the folder where the initial files for using the face tracking algorithms are located must be specified (they should be in this repository).

### 1.1 Example of use:

```
loadingPath = r'J:\Original_Datasets\MMSE-HR\MMSE-HR' # Path where we can find original files
savingPath = r'J:\faces\128_128\original\MMSE' # Path where faces will be saved
filespath = r'E:\repos\face_cropper\source' #Path with the files needed to run this script
MMSE = face_cropper_MMSE(loadingPath, savingPath, filespath, SHOW=False)
MMSE.find_files()
if not(MMSE.same_number_of_files()):
    print('Error: pathFiles and pathGT must have the same number of files')
    sys.exit()
else:
    MMSE.copy_GT_files()
    MMSE.crop_faces((128,128))
```
### 1.1.1 Explanation

`MMSE = face_cropper_MMSE(loadingPath, savingPath, filespath, SHOW=False)`: Initialization

`MMSE.find_files()`: Creates "self.datapath" with the paths of all input videos, and "self.gtpath" with the paths to all ground truth files.

`MMSE.copy_GT_files`: copies the ground truth files from the original database, renames them with the subject name and ending in _gt, and pastes them into "savingPath". For example, for input subject F0013 the ground truth file "BP_mm.txt", the output will be the same file but renamed as "F0013_gt.txt". (In VIPL the timestamp file is also saved following the same process but with the ending _timestamp Ex: "F0013_timestamp.txt").

`MMSE.crop_faces((128,128))`:is the main function of this routine. Specifically, the neural network "readNetFromCaffe" is used to detect the face in the scene. Then the face is tracked with the method "cv2.TrackerMedianFlow_create". If the face is lost, it is searched again in the next frame. If no face is found in the scene, an image is saved in zeros (black image) to conserve the number of frames. The (128,128) part refers to the output size of the faces in pixels.

### 1.2 Outputs
![image2](https://drive.google.com/uc?export=view&id=1OwC9MEFVpZgKUTcHO0btrvZ90jrKzIWe)
"savingPath" will have one folder per subject with his respectively name and one file called "Dataset_Report.txt" with global information about all subjects. Inside each subject folder, there will be the .png files with the faces cropped, the ground truth file renamed (and timestamp if it is the case), and a individual "Report.txt" for the specific subject.