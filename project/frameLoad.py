import cv2
import numpy as np
import os
# class VideoToFramesConverter():

def convert_video_to_frames(video_dir, frames_dir):
    try:
        os.mkdir(frames_dir)
    except OSError:
        pass
    vidcap = cv2.VideoCapture(video_dir)
    #vidcap.set(cv2.CV_CAP_PROP_FPS, 100)

    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      cv2.imwrite(frames_dir +"/frame%d.jpg" % count, image)     # save frame as JPEG file
      print "save frame as "+ frames_dir +"/frame%d.jpg" % count
      success,image = vidcap.read()
      print 'Read a new frame: ', success
      count += 1

    vidcap.release()
    return


def get_first_k_frames(frames_dir, k):
    frames = []
    try:
        for i in range(k):
            img = cv2.imread(frames_dir +"/frame%d.jpg" % i)
            if img == None:
                return frames
            frames.append(img)
    except:
        return frames
    return frames


def main():

    video_dir = "/Users/macbook/Documents/CS231A/project/automatic-piano-transcription/project/wei_last.mp4"
    frames_dir = "/Users/macbook/Documents/CS231A/project/automatic-piano-transcription/project/data"
    convert_video_to_frames(video_dir, frames_dir)
    frames = get_first_k_frames(frames_dir, 5)
    print len(frames)
    '''
    # # img = cv2.imread('data/arjun1.jpg')
    # # img = cv2.imread('keyboard-2.jpg')
    # img = cv2.imread('testimage2.jpg')
    # # detectKeyboard(img)
    # # readVideo('vid-3.mp4')
    # # img = cv2.imread('frame-3.jpg')
    # # img = cv2.imread('img.png')
    #
    # # img = cv2.imread('img.png')
    # # detect_black_keys(img)
    #
    # #img = cv2.imread('currFrame.jpg')
    # # img = cv2.imread('frame-3.jpg')
    # img = cv2.imread('bg.jpg')
    # keyboard = processedFrame(img)
    # if keyboard is not None:
    #     display_image(keyboard)
    # cv2.imwrite("background"+".jpg", keyboard)
    # readVideo('vid\\sample-jazz-tut-1.mp4')

    # bgd = cv2.imread('bg.jpg')
    # frame = cv2.imread('currFrame.jpg')

    # gray = cv2.cvtColor(bgd, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # bgd_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # frame_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # positive_diff = frame_thresh-bgd_thresh
    # positive_diff = (positive_diff > 0) * positive_diff
    # # display_image(positive_diff)
    # cv2.imwrite('positive_diff.jpg',positive_diff)

    # gray = cv2.cvtColor(bgd, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # bgd_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # frame_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # negative_diff = frame_thresh-bgd_thresh
    # negative_diff = (negative_diff > 0) * negative_diff
    # # display_image(negative_diff)
    # cv2.imwrite('negative_diff.jpg', negative_diff)

    return
'''
if __name__ == '__main__':
	main()
