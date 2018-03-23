import numpy as np
import cv2
from frameLoad import *
from keyBoardDetection import detect_keyboard
from keyDetection import *
from NoteDetection import *


# class PianoTranscriptor():
#     v2f = VideoToFramesConverter()
#     v2f.load_video_to_frames()

# convert_video_to_frames(video_dir, frames_dir)
# images = get_first_k_frames(frames_dir, k = 5 )


# background_img = images[0]
background_img= cv2.imread('background.jpg')
keyboard_img = cv2.imread('keyboard.jpg')
black_key_properties = detect_black_keys(background_img)
black_keys_pattern, black_notes = assign_black_keys(black_key_properties)
upper_white_properties, lower_white_properties = detect_white_keys(background_img, black_key_properties, black_keys_pattern, black_notes)
white_keys_pattern_right_of_black, white_notes = assign_white_keys(black_keys_pattern, upper_white_properties)

for i in range(1, len(images)):
    keyboard_img = images[i]
    keyboard = detect_keyboard(keyboard_img)
    best_note_idx_guess, black_or_white = detect_note(keyboard_img, background_img, black_key_properties, upper_white_properties)
    best_note_guess = assign_best_note_guess(best_note_idx_guess, black_or_white, black_notes, white_notes)
    print best_note_guess
