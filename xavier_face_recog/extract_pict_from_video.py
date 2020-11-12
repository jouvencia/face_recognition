import cv2
import os
video_dir='./Desktop/Boulot/JouvencIA/videos_residents'
for root, dirs, files in os.walk(video_dir):
    for file in files:
        path=os.path.join(root,file)
        name=os.path.splitext(file)[0]
        try:
            os.mkdir(video_dir + '/../photos_residents/train/' + name)
        except:
            print('eh mince')

# Opens the Video file
        cap= cv2.VideoCapture(path)
        i=0
        while(cap.isOpened()):
            ret, frame = cap.read()
            #frame=cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            if ret == False:
                break

            if i%5==0:
                cv2.imwrite(video_dir + '/../photos_residents/train/' + name + '/' + name + '_' + str(i)+'.jpg',frame)
            
            i+=1
        
        cap.release()
        cv2.destroyAllWindows()