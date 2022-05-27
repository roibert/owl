import cv2


def get_model_name(number_conv, number_dense, batch_size, model_suffix=""):

    model_name = "m_"

    model_name = "%s%d_conv_%d_dense_%d_batch" % (model_name, number_conv, number_dense, batch_size)

    model_name += model_suffix

    return model_name


def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    image = (1 / 255.) * image

    return image


def get_image_from_frame_name(frame, frame_folder):
    frame_path = ("%s/%s" % (frame_folder, frame))

    return load_image(frame_path)


def create_gap_in_set(set_to_modify, lower_limit, upper_limit, test_set, key):
    """
    Creates a gap in the 'set_to_modify' in the range ['lower_limit', 'upper_limit'].
    The data points which have been removed will be added to 'test_set'

    :param set_to_modify:
    :param lower_limit:
    :param upper_limit:
    :param test_set:
    :return: set_to_set_to_modify, test_set
    """

    # add excluded area
    test_set.append(
        set_to_modify[
            (set_to_modify[key] >= lower_limit) & (set_to_modify[key] <= upper_limit)
            ]
    )

    # remove excluded area
    set_to_modify = \
        set_to_modify[
            (set_to_modify[key] < lower_limit) | (set_to_modify[key] > upper_limit)
            ]

    return set_to_modify, test_set
