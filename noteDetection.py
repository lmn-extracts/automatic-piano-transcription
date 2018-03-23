import numpy as np
import cv2
import imutils
from keyDetection import *


def hand_detection(bin_wkeys):
    # find outter bounding boxes
    #grey_bin_wkeys=cv2.cvtColor(bin_wkeys,cv2.COLOR_BGR2GRAY)
    #contours = cv2.findContours(grey_bin_wkeys,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(bin_wkeys, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lst_hands= []

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        # the bounding box contains hands should larger than some MinWidth, say = black_key_height/10
        if w < 2*bin_wkeys.shape[0]/3./10.:
            continue
        lst_hands.append((x,y,w,h))
    # print "lst_hands", lst_hands
    return lst_hands

def key_inside_hand(key_property, lst_hands):
    x,y,w,h = key_property
    #print "key_property", key_property

    for hand in lst_hands:
        handx, handw = hand[0], hand[3]
        #print "hand", hand
        if x>handx and x+w < handx+handw:
            return True
    return False

def dectect_blob(bin_keys, key_property):
    (x,y,w,h) = key_property
    # print "key_property in dectect_blob", key_property
    # print "bin_keys", bin_keys.shape
    extracted_bin_wkeys = bin_keys[y:y+h, x:x+w]
    # cv2.imshow('frame', extracted_bin_wkeys)
    # cv2.waitKey(0)
    #grey_bin_wkeys=cv2.cvtColor(extracted_bin_wkeys,cv2.COLOR_BGR2GRAY)
    _, contours, hierarchy = cv2.findContours(extracted_bin_wkeys, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs_area_sum = 0
    for cnt in contours:
        blobs_area_sum += cv2.contourArea(cnt)
        cnt = cnt.astype("int")
    #     cv2.drawContours(bin_keys, [cnt], -1, (0, 255, 0), 2)
    #     cv2.fillPoly(bin_keys, [cnt], (0, 255, 0))
    #     cv2.imshow('bin_keys_cnt',bin_keys)
    #     cv2.waitKey(0)
    # print "len(contours)", len(contours)
    return blobs_area_sum

def detect_note(keyboard_img, background_img, black_key_properties, upper_white_properties):
    # h, w, rgb = keyboard_img.shape
    # kb_subtract_bg = np.zeros((h,w,rgb), dtype=int)
    # bg_subtract_kb = np.zeros((h,w,rgb), dtype=int)
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(rgb):
    #             if keyboard_img[i,j,k] > background_img[i,j,k]:
    #                 kb_subtract_bg[i,j,k] = keyboard_img[i,j,k] - background_img[i,j,k]
    #             else:
    #                 bg_subtract_kb[i,j,k] = background_img[i,j,k] - keyboard_img[i,j,k]
    #
    # print "keyboard_img", keyboard_img[55, 305,:]
    # print "background_img", background_img[55, 305,:]
    # print "diff", kb_subtract_bg[55, 305,:]
    # print "diff1", bg_subtract_kb[55, 305,:]
    #
    # cv2.imshow('kb_subtract_bg',kb_subtract_bg)
    # cv2.waitKey(0)
    #
    # print "keyboard_img1", keyboard_img.shape, type(keyboard_img)
    # cv2.imshow('keyboard_img1',keyboard_img)
    # cv2.waitKey(0)
    # print "background_img1", background_img.shape, type(background_img)
    # cv2.imshow('background_img1',background_img)
    # cv2.waitKey(0)
    # kb_subtract_bg = keyboard_img - background_img
    # print "kb_subtract_bg", kb_subtract_bg[:2, :,:]
    #
    #
    # cv2.imshow('kb_subtract_bg',kb_subtract_bg)
    # cv2.waitKey(0)
    # bg_subtract_kb = -kb_subtract_bg
    # cv2.imshow('bg_subtract_kb',bg_subtract_kb)
    # cv2.waitKey(0)



    # # normalize
    # norm_kb_subtract_bg = np.copy(kb_subtract_bg)
    # cv2.normalize(kb_subtract_bg, norm_kb_subtract_bg, 0, 255, cv2.NORM_MINMAX)
    # #print "kb_subtract_bg?", kb_subtract_bg
    # #print "norm_kb_subtract_bg?", norm_kb_subtract_bg
    # print "min", np.min(norm_kb_subtract_bg), "max", np.max(norm_kb_subtract_bg)
    # norm_bg_subtract_kb = np.copy(bg_subtract_kb)
    # cv2.normalize(bg_subtract_kb, norm_bg_subtract_kb, 0, 255, cv2.NORM_MINMAX)
    #binarize
    grey_keyboard_img = cv2.cvtColor(keyboard_img,cv2.COLOR_BGR2GRAY)
    grey_background_img = cv2.cvtColor(background_img,cv2.COLOR_BGR2GRAY)

    h, w, rgb = keyboard_img.shape
    kb_subtract_bg = np.zeros((h,w), dtype=np.uint8)
    bg_subtract_kb = np.zeros((h,w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if grey_keyboard_img[i,j] > grey_background_img[i,j]:
                kb_subtract_bg[i,j] = grey_keyboard_img[i,j] - grey_background_img[i,j]
            else:
                bg_subtract_kb[i,j] = grey_background_img[i,j] - grey_keyboard_img[i,j]

    # # normalize
    # norm_kb_subtract_bg = np.copy(kb_subtract_bg)
    # cv2.normalize(kb_subtract_bg, norm_kb_subtract_bg, 0, 255, cv2.NORM_MINMAX, 1)
    # #print "kb_subtract_bg?", kb_subtract_bg
    # #print "norm_kb_subtract_bg?", norm_kb_subtract_bg
    # print "min", np.min(norm_kb_subtract_bg), "max", np.max(norm_kb_subtract_bg)
    # norm_bg_subtract_kb = np.copy(bg_subtract_kb)
    # cv2.normalize(bg_subtract_kb, norm_bg_subtract_kb, 0, 255, cv2.NORM_MINMAX, 1)


    # print "grey_keyboard_img", grey_keyboard_img[-1,:]
    # print "grey_background_img", grey_background_img[-1,:]
    # print "kb_subtract_bg?", kb_subtract_bg[-1,:]
    # print "norm_kb_subtract_bg?", norm_kb_subtract_bg[-1,:]
    
    # cv2.imshow('kb_subtract_bg', kb_subtract_bg)
    # cv2.waitKey(0)

    # norm_kb_subtract_bg = grey_keyboard_img
    # norm_bg_subtract_kb = grey_background_img
    #
    # grey_wkeys=cv2.cvtColor(norm_kb_subtract_bg,cv2.COLOR_BGR2GRAY)
    # grey_bkeys=cv2.cvtColor(norm_bg_subtract_kb,cv2.COLOR_BGR2GRAY)
    #
    #
    thres_wkeys = np.percentile(bg_subtract_kb[bg_subtract_kb > 0], 90)
    # print "bg_subtract_kb", np.percentile(bg_subtract_kb[bg_subtract_kb > 0], 90)
    retval_w, bin_wkeys = cv2.threshold(bg_subtract_kb, thres_wkeys, 255, cv2.THRESH_BINARY)

    thres_bkeys = np.percentile(kb_subtract_bg[kb_subtract_bg > 0], 90)
    retval_b, bin_bkeys = cv2.threshold(kb_subtract_bg, thres_bkeys, 255, cv2.THRESH_BINARY)

    # cv2.imshow('bin_wkeys', bin_wkeys)
    # cv2.waitKey(0)
    # cv2.imshow('bin_bkeys', bin_bkeys)
    # cv2.waitKey(0)



    # print "bin_bkeys?", bin_bkeys.shape
    #detect areas of hands
    lst_hands = hand_detection(bin_wkeys);
    #check pressed key can be a black key
    num_black_keys = len(black_key_properties)
    blobs_in_kb_img = []
    # default is no keystroke
    max_blobs_area_sum = 0
    best_note_guess = -1
    black_or_white = None
    # assume only one keystroke at a time, so find the blob with max area
    for i in range(num_black_keys):
        if key_inside_hand(black_key_properties[i], lst_hands):
            # print "i_num_black_keys, inside hands", i
            blobs_area_sum = dectect_blob(bin_bkeys, black_key_properties[i])
            # print "blobs_area_sum", blobs_area_sum
            if blobs_area_sum > max_blobs_area_sum:
                max_blobs_area_sum = blobs_area_sum
                best_note_guess = i
                black_or_white = "b"
    #check pressed key can be a white key
    # print "?"
    num_white_keys = len(upper_white_properties)
    for i in range(num_white_keys):
        if key_inside_hand(upper_white_properties[i], lst_hands):
            # print "i_num_white_keys", i
            blobs_area_sum = dectect_blob(bin_wkeys, upper_white_properties[i])
            # print "blobs_area_sum", blobs_area_sum
            if blobs_area_sum > max_blobs_area_sum:
                max_blobs_area_sum = blobs_area_sum
                best_note_guess = i
                black_or_white = "w"
    # print "max_blobs_area_sum", max_blobs_area_sum
    # print "best_note_guess", best_note_guess
    # print "black_or_white", black_or_white
    return best_note_guess, black_or_white


def detect_notes_in_frame(keyboard_img, background_img):
    black_key_properties = detectKeys(background_img)
    pattern = assign_white_keys(black_key_properties)
    upper_white_properties, lower_white_properties, white_notes, black_notes = detect_white_keys(background_img, black_key_properties, pattern)
    best_note_guess, black_or_white = detect_note(keyboard_img, background_img, black_key_properties, upper_white_properties)
    if black_or_white == 'w':
        return white_notes[best_note_guess], highlight_key(keyboard_img, best_note_guess, black_or_white, black_key_properties, upper_white_properties, lower_white_properties)
    else:
        return black_notes[best_note_guess]

def highlight_key(img, best_note_guess, black_or_white, black_key_properties, upper_white_properties, lower_white_properties):
    if black_or_white == 'w':
        x,y,w,h = upper_white_properties[best_note_guess]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),1)
        x,y,w,h = lower_white_properties[best_note_guess]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),1)
        cv2.imwrite('highlight_key.jpg', img)
    else:
        x,y,w,h = black_key_properties[best_note_guess]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),1)
    return img


def main():
    background_img= cv2.imread('background.jpg')
    keyboard_img = cv2.imread('keyboard.jpg')
    # keyboard_img = cv2.imread('frame141.jpg')
    black_key_properties = detectKeys(background_img)
    # print "detectKeys black_key_properties", black_key_properties
    pattern = assign_white_keys(black_key_properties)
    # print "pattern", pattern
    upper_white_properties, lower_white_properties, white_notes, black_notes = detect_white_keys(background_img, black_key_properties, pattern)
    best_note_guess, black_or_white = detect_note(keyboard_img, background_img, black_key_properties, upper_white_properties)
    # print best_note_guess   
    # print black_or_white
    print white_notes
    print best_note_guess
    print white_notes[best_note_guess]
    print len(upper_white_properties)
    print len(lower_white_properties)
    highlight_key(keyboard_img, best_note_guess, black_or_white, black_key_properties, lower_white_properties, upper_white_properties)
if __name__ == "__main__":
    main()
