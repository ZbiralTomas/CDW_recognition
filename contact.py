from rembg import remove
import numpy as np
import cv2
import os.path
import time

vertical_line_start = 500
vertical_line_end = 100
vertical_line_save = 290
threshold = 1500
part = 1
image_width = 640
image_height = 480
box_color = (0, 0, 255)
box_thickness = 5
box_left = 0
box_top = 0
box_right = image_width - 1
box_bottom = image_height - 1
delay_duration = 0.2
j = 0
output_filename = "video_conference.mp4"
frame_width = 640
frame_height = 480
fps = 45
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))



def is_contact(prev, curr, part, threshold, line):
    height = prev.shape[0]
    lower_limit = round((1 - part) / 2 * height)
    upper_limit = round((1 - (1 - part) / 2) * height)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    diff_sum = np.sum(frame_diff[lower_limit:upper_limit, line])
    return diff_sum >= threshold


prev_frame = cv2.imread("videos/updated_video/frame120.jpg")
i = 121
is_file = True
end_contact = True
mid_contact = True

while is_file:
    frame_filename = f"frame{i:d}.jpg"
    current_frame = cv2.imread("videos/updated_video/" + frame_filename)
    i += 1
    is_file = os.path.isfile("videos/updated_video/" + f"frame{i:d}.jpg")
    frame = current_frame.copy()
    frame = cv2.line(frame, (vertical_line_start, 0), (vertical_line_start,
                                                       current_frame.shape[0]), (0, 255, 0), 2)
    frame = cv2.line(frame, (vertical_line_end, 0), (vertical_line_end,
                                                     current_frame.shape[0]), (0, 0, 255), 2)
    if i % 100 == 0:
        print(i)
    if end_contact:
        if is_contact(prev_frame, current_frame, part, threshold, vertical_line_start):
            print("Contact at the start")
            end_contact = False
            mid_contact = False
            contact_image = cv2.line(current_frame, (vertical_line_start, 0), (vertical_line_start,
                                                                               current_frame.shape[0]), (0, 255, 0), 2)
            cv2.imwrite("videos/start_contact/" + frame_filename, contact_image)
    else:
        if is_contact(prev_frame, current_frame, part, threshold, vertical_line_end):
            print("Contact at the end")
            end_contact = True
            contact_image = cv2.line(current_frame, (vertical_line_end, 0), (vertical_line_end,
                                                                             current_frame.shape[0]), (0, 255, 0), 2)
            cv2.imwrite("videos/end_contact/" + frame_filename, contact_image)
    if end_contact:
        frame = cv2.circle(frame, (600, 450), 20, (0, 255, 0), -1)
    else:
        frame = cv2.circle(frame, (600, 450), 20, (0, 0, 255), -1)
        if is_contact(prev_frame, current_frame, part, threshold, vertical_line_save) and not mid_contact:
            cv2.rectangle(
                frame,
                (box_left, box_top),
                (box_right, box_bottom),
                box_color,
                box_thickness,
            )
            j += 1
        if j == 5:
            if is_contact(prev_frame, current_frame, part, threshold, vertical_line_save):
                mid_contact = True
                j = 0

    out.write(frame)

    if i % 4 == 0:
        prev_frame = current_frame




    """
    removed = remove(frame)
    removed_np = np.array(removed)
    mask = np.moveaxis(removed_np, -1, 0)
    mask = mask[3]
    mask.setflags(write=1)
    mask[mask < 100] = 0
    mask[mask >= 100] = 1
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    im_v = cv2.vconcat([frame, masked])
    cv2.imshow('ImageWindow', im_v)
    """

out.release()
