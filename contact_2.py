from rembg import remove
import numpy as np
import cv2
import os.path

vertical_line_start = 500
vertical_line_end = 100
threshold = 1600
part = 0.8


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

while is_file:
    frame_filename = f"frame{i:d}.jpg"
    current_frame = cv2.imread("videos/updated_video/" + frame_filename)
    i += 1
    is_file = os.path.isfile("videos/updated_video/" + f"frame{i:d}.jpg")
    frame = cv2.line(current_frame, (vertical_line_start, 0), (vertical_line_start,
                                                               current_frame.shape[0]), (0, 255, 0), 2)
    frame = cv2.line(frame, (vertical_line_end, 0), (vertical_line_end,
                                                     current_frame.shape[0]), (0, 0, 255), 2)
    if i % 100 == 0:
        print(i)
    if end_contact:
        frame = cv2.circle(frame, (450, 600), 14, (0, 255, 0), -1)
        frame = cv2.circle(frame, (450, 560), 14, (0, 0, 255), 2)
        if is_contact(prev_frame, current_frame, part, threshold, vertical_line_start):
            print("Contact at the start")
            end_contact = False
            contact_image = cv2.line(current_frame, (vertical_line_start, 0), (vertical_line_start,
                                                                               current_frame.shape[0]), (0, 255, 0), 2)
            cv2.imwrite("videos/start_contact/" + frame_filename, contact_image)
    else:
        frame = cv2.circle(frame, (450, 600), 14, (0, 255, 0), 2)
        frame = cv2.circle(frame, (450, 560), 14, (0, 0, 255), -1)
        if is_contact(prev_frame, current_frame, part, threshold, vertical_line_end):
            print("Contact at the end")
            end_contact = True
            contact_image = cv2.line(current_frame, (vertical_line_end, 0), (vertical_line_end,
                                                                             current_frame.shape[0]), (0, 255, 0), 2)
            cv2.imwrite("videos/end_contact/" + frame_filename, contact_image)
    prev_frame = current_frame
    cv2.imshow("window", frame)

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
