import random
import cv2

def opencv_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = 'E:/Lip_Code/Web_data/uploads/{}.mp4'.format(random.randint(0,10000000))
    out = cv2.VideoWriter(video_path,fourcc,20.0,(640,480))
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame,2)
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(2) == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return video_path


if __name__ == '__main__':
    opencv_video()

