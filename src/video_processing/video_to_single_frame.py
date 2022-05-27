import re
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join

import cv2

number_pattern = r"[0-9]+"

def _get_one_number_from_string(string):

    number = re.findall(number_pattern, string)
    return int(number[0])


def _get_video_number(name):

    vide_key = r"__v\d+__"
    video_match = re.findall(vide_key, name)

    if not video_match:
        return None

    return _get_one_number_from_string(video_match[0])


def _get_video_timestamp(video):

    date_extractor = r"2021_\d\d_\d\d_\d\d_\d\d_\d\d"
    date_match = re.findall(date_extractor, video)
    date_str = date_match[0]

    return datetime.strptime(date_str, "%Y_%m_%d_%H_%M_%S")


def video_to_single_frame(  video_folder_path,
                            run_identifier,
                            output_path,
                            cropping_coordinates=None,
                            image_suffix=".jpg"):

    all_files = [f for f in listdir(video_folder_path) if ( isfile(join(video_folder_path, f))
                                                            and ("h264" in f)
                                                            and (run_identifier in f)
                                                            and not ("_backup" in f)
                                                            )]

    def _get_key_from_video_name(name):

        date_extractor = r"__15__25_\S+"
        match = re.findall(date_extractor, name)
        return match[0]

    all_files.sort(key=lambda name: _get_key_from_video_name(name))

    n = len(all_files)
    file_count = 1

    for file in all_files:

        print("Converting file %d/%d." % (file_count, n))
        file_count += 1

        video = "%s/%s" % (video_folder_path, file)
        cap = cv2.VideoCapture(video)

        if not cap.isOpened():
            raise ValueError("Video not opened.")

        video_number = _get_video_number(file)

        frames_per_second = float(cap.get(cv2.CAP_PROP_FPS))
        time_per_frame = timedelta(milliseconds=1000/frames_per_second)
        current_frame = 0

        current_timestamp = _get_video_timestamp(file)

        while True:
            flag, frame = cap.read()

            if not flag:
                break

            # crop image
            # width:
            if not cropping_coordinates is None:
                y = cropping_coordinates[0]
                h = cropping_coordinates[1]

                # height
                x = cropping_coordinates[2]
                w = cropping_coordinates[3]

                cropped_image = frame[x:w, y:h]
            else:
                cropped_image = frame

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            cropped_image = cv2.Canny(cropped_image, 60, 120)

            height = cropped_image.shape[0]
            width = cropped_image.shape[1]

            current_frame += 1
            current_timestamp += time_per_frame

            frame_number_zero_padding_max_length = 4
            zero_padded_frame_number = str(current_frame).zfill(frame_number_zero_padding_max_length)
            frame_timestamp = current_timestamp.strftime("%Y_%m_%d_%H_%M_%S_%f")

            frame_output_path = "%s/%s_v%d_%d_%d_f%s_%s%s" \
                                % (output_path,
                                   run_identifier,
                                   video_number,
                                   height,
                                   width,
                                   zero_padded_frame_number,
                                   frame_timestamp,
                                   image_suffix)

            cv2.imwrite(frame_output_path, cropped_image)

        cap.release()
