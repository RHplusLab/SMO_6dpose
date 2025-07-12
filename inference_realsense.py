from curses import BUTTON1_CLICKED
import cv2
import numpy as np
import os
import math

import pyrealsense2 as rs
import tensorflow as tf

from model import build_Model
from utils import preprocess_image
from utils.visualization import draw_detections


def main():
    """
    Run Dense6DPose in inference mode live on Realsense camera.
    
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()

    #input parameter
    phi = 0
    path_to_weights = "weights\\ycbv_best.h5"
    
    # save_path = "./predictions/occlusion/" #where to save the images or None if the images should be displayed and not saved
    save_path = None
    image_extension = ".jpg"
    # YCB-Video 데이터셋의 21개 객체
    class_to_name = {
        0: "002_master_chef_can", 1: "003_cracker_box", 2: "004_sugar_box", 
        3: "005_tomato_soup_can", 4: "006_mustard_bottle", 5: "007_tuna_fish_can",
        6: "008_pudding_box", 7: "009_gelatin_box", 8: "010_potted_meat_can",
        9: "011_banana", 10: "019_pitcher_base", 11: "021_bleach_cleanser",
        12: "024_bowl", 13: "025_mug", 14: "035_power_drill", 
        15: "036_wood_block", 16: "037_scissors", 17: "040_large_marker",
        18: "051_large_clamp", 19: "052_extra_large_clamp", 20: "061_foam_brick"
    }
    #class_to_name = {0: "driller"} #Linemod use a single class with a name of the Linemod objects
    score_threshold = 0.5
    translation_scale_norm = 1000.0
    draw_bbox_2d = False
    draw_name = False
    #you probably need to replace the realsense camera matrix with the one of your webcam
    camera_matrix = get_realsense_camera_matrix()
    name_to_3d_bboxes = get_ycbvideo_3d_bboxes()
    class_to_3d_bboxes = {class_idx: name_to_3d_bboxes[name] for class_idx, name in class_to_name.items()} 
    
    num_classes = len(class_to_name)
    
    #build model and load weights
    model, image_size = build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights)
    
    # Initialize Realsense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(1280, 960, rs.format.bgr8, 30)
    pipeline.start(config)
    
    #inferencing
    print("\nStarting inference...\n")
    i = 0
    try:
        while True:
            #load image
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frames()
            
            if not color_frame:
                continue
            
            # Realsense frame to numpy array
            image = np.asanyarray(color_frame.get_data())
            original_image = image.copy()
            
            #preprocessing
            input_list, scale = preprocess(image, image_size, camera_matrix, translation_scale_norm)
            
            #predict
            boxes, scores, labels, rotations, translations = model.predict_on_batch(input_list)
            
            #postprocessing
            boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold)
            
            draw_detections(original_image,
                            boxes,
                            scores,
                            labels,
                            rotations,
                            translations,
                            class_to_bbox_3D = class_to_3d_bboxes,
                            camera_matrix = camera_matrix,
                            label_to_name = class_to_name,
                            draw_bbox_2d = draw_bbox_2d,
                            draw_name = draw_name)
            
            #display image with predictions
            cv2.imshow('image with predictions', original_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if not save_path is None:
                #images to the given path
                os.makedirs(save_path, exist_ok = True)
                cv2.imwrite(os.path.join(save_path, "frame_{}".format(i) + image_extension), original_image)
                
            i += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    
def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # 동적 메모리 할당
            print(len(gpus), "Physical GPUs,", len(tf.config.list_logical_devices('GPU')), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def get_realsense_camera_matrix():
    """
    Returns:
        The Linemod and Occlusion 3x3 camera matrix

    """
    fx, fy = 638.956, 638.394
    px, py = 630.634, 367.101
    return np.array([[fx, 0., px],
                        [0., fy, py],
                        [0., 0., 1.]], dtype=np.float32)

def get_ycbvideo_3d_bboxes():
    """
    Returns:
        name_to_3d_bboxes: Dictionary with the Linemod and Occlusion 3D model names as keys and the cuboids as values

    """
    name_to_model_info = {
        "002_master_chef_can": {"diameter": 104.0, "min_x": -52.0, "min_y": -52.0, "min_z": -100.0, "size_x": 104.0, "size_y": 104.0, "size_z": 200.0},
        "003_cracker_box": {"diameter": 210.0, "min_x": -80.0, "min_y": -50.0, "min_z": -100.0, "size_x": 160.0, "size_y": 100.0, "size_z": 200.0},
        "004_sugar_box": {"diameter": 180.0, "min_x": -50.0, "min_y": -90.0, "min_z": -100.0, "size_x": 100.0, "size_y": 180.0, "size_z": 200.0},
        "005_tomato_soup_can": {"diameter": 102.0, "min_x": -51.0, "min_y": -51.0, "min_z": -100.0, "size_x": 102.0, "size_y": 102.0, "size_z": 200.0},
        "006_mustard_bottle": {"diameter": 190.0, "min_x": -60.0, "min_y": -60.0, "min_z": -95.0, "size_x": 120.0, "size_y": 120.0, "size_z": 190.0},
        "007_tuna_fish_can": {"diameter": 85.0, "min_x": -42.5, "min_y": -42.5, "min_z": -50.0, "size_x": 85.0, "size_y": 85.0, "size_z": 100.0},
        "008_pudding_box": {"diameter": 150.0, "min_x": -50.0, "min_y": -70.0, "min_z": -75.0, "size_x": 100.0, "size_y": 140.0, "size_z": 150.0},
        "009_gelatin_box": {"diameter": 140.0, "min_x": -45.0, "min_y": -70.0, "min_z": -70.0, "size_x": 90.0, "size_y": 140.0, "size_z": 140.0},
        "010_potted_meat_can": {"diameter": 100.0, "min_x": -50.0, "min_y": -50.0, "min_z": -100.0, "size_x": 100.0, "size_y": 100.0, "size_z": 200.0},
        "011_banana": {"diameter": 150.0, "min_x": -30.0, "min_y": -30.0, "min_z": -75.0, "size_x": 60.0, "size_y": 60.0, "size_z": 150.0},
        "019_pitcher_base": {"diameter": 200.0, "min_x": -80.0, "min_y": -80.0, "min_z": -100.0, "size_x": 160.0, "size_y": 160.0, "size_z": 200.0},
        "021_bleach_cleanser": {"diameter": 250.0, "min_x": -70.0, "min_y": -70.0, "min_z": -125.0, "size_x": 140.0, "size_y": 140.0, "size_z": 250.0},
        "024_bowl": {"diameter": 150.0, "min_x": -75.0, "min_y": -75.0, "min_z": -50.0, "size_x": 150.0, "size_y": 150.0, "size_z": 100.0},
        "025_mug": {"diameter": 120.0, "min_x": -60.0, "min_y": -60.0, "min_z": -60.0, "size_x": 120.0, "size_y": 120.0, "size_z": 120.0},
        "035_power_drill": {"diameter": 200.0, "min_x": -100.0, "min_y": -50.0, "min_z": -100.0, "size_x": 200.0, "size_y": 100.0, "size_z": 200.0},
        "036_wood_block": {"diameter": 150.0, "min_x": -75.0, "min_y": -25.0, "min_z": -75.0, "size_x": 150.0, "size_y": 50.0, "size_z": 150.0},
        "037_scissors": {"diameter": 180.0, "min_x": -90.0, "min_y": -20.0, "min_z": -90.0, "size_x": 180.0, "size_y": 40.0, "size_z": 180.0},
        "040_large_marker": {"diameter": 100.0, "min_x": -20.0, "min_y": -20.0, "min_z": -50.0, "size_x": 40.0, "size_y": 40.0, "size_z": 100.0},
        "051_large_clamp": {"diameter": 150.0, "min_x": -75.0, "min_y": -30.0, "min_z": -75.0, "size_x": 150.0, "size_y": 60.0, "size_z": 150.0},
        "052_extra_large_clamp": {"diameter": 200.0, "min_x": -100.0, "min_y": -40.0, "min_z": -100.0, "size_x": 200.0, "size_y": 80.0, "size_z": 200.0},
        "061_foam_brick": {"diameter": 120.0, "min_x": -60.0, "min_y": -40.0, "min_z": -60.0, "size_x": 120.0, "size_y": 80.0, "size_z": 120.0}
    }        
    name_to_3d_bboxes = {name: convert_bbox_3d(model_info) for name, model_info in name_to_model_info.items()}
    
    return name_to_3d_bboxes


def convert_bbox_3d(model_dict):
    """
    Converts the 3D model cuboids from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.

    """
    #get infos from model dict
    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]
    
    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]
    
    bbox = np.zeros(shape = (8, 3))
    #lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    #upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
    
    return bbox


def build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights):
    """
    Builds an a model and init it with a given weight file
    Args:
        phi: scaling hyperparameter
        num_classes: The number of classes
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        path_to_weights: Path to the weight file
        
    Returns:
        model_prediction: The model
        image_size: Integer image size used as the model input resolution for the given phi

    """
    print("\nBuilding model...\n")
    _, model_prediction = build_Model(phi,                                                                                                                          
                                                         num_classes = num_classes,
                                                         num_anchors = 9,
                                                         freeze_bn = True,
                                                         score_threshold = score_threshold,
                                                         num_rotation_parameters = 3,
                                                         backbone = 'densenet',
                                                         print_architecture = False)
    
    print("\nDone!\n\nLoading weights...")
    model_prediction.load_weights(path_to_weights, by_name=True)
    print("Done!")
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    
    return model_prediction, image_size


def preprocess(image, image_size, camera_matrix, translation_scale_norm):
    """
    Preprocesses the inputs for the model
    Args:
        image: The image to predict
        image_size: Input resolution for the model
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        translation_scale_norm: factor to change units. model internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_list: List containing the preprocessed inputs for the model
        scale: The scale factor of the resized input image and the original image

    """
    image = image[:, :, ::-1]
    image, scale = preprocess_image(image, image_size)
    camera_input = get_camera_parameter_input(camera_matrix, scale, translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    return input_list, scale


def get_camera_parameter_input(camera_matrix, image_scale, translation_scale_norm):
    """
    Return the input vector for the camera parameter layer
    Args:
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        image_scale: The scale factor of the resized input image and the original image
        translation_scale_norm: factor to change units. the model internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_vector: numpy array [fx, fy, px, py, translation_scale_norm, image_scale]

    """
    #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
    input_vector = np.zeros((6,), dtype = np.float32)
    
    input_vector[0] = camera_matrix[0, 0]
    input_vector[1] = camera_matrix[1, 1]
    input_vector[2] = camera_matrix[0, 2]
    input_vector[3] = camera_matrix[1, 2]
    input_vector[4] = translation_scale_norm
    input_vector[5] = image_scale
    
    return input_vector


def postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold):
    """
    Filter out detections with low confidence scores and rescale the outputs of the model
    Args:
        boxes: numpy array [batch_size = 1, max_detections, 4] containing the 2D bounding boxes
        scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
        labels: numpy array [batch_size = 1, max_detections] containing class label
        rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
        scale: The scale factor of the resized input image and the original image
        score_threshold: Minimum score threshold at which a prediction is not filtered out
    Returns:
        boxes: numpy array [num_valid_detections, 4] containing the 2D bounding boxes
        scores: numpy array [num_valid_detections] containing the confidence scores
        labels: numpy array [num_valid_detections] containing class label
        rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [num_valid_detections, 3] containing the translation vectors

    """
    boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
    # correct boxes for image scale
    boxes /= scale
    #rescale rotations
    rotations *= math.pi
    #filter out detections with low scores
    indices = np.where(scores[:] > score_threshold)
    # select detections
    scores = scores[indices]
    boxes = boxes[indices]
    rotations = rotations[indices]
    translations = translations[indices]
    labels = labels[indices]
    
    return boxes, scores, labels, rotations, translations


if __name__ == '__main__':
    main()
