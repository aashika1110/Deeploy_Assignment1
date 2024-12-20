import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def run_mask_rcnn(image_path):
    """
    Run a pre-trained Mask R-CNN model on the input image to detect objects.
    :param image_path: Path to the input image
    """
    # Step 1: Load the pre-trained Mask R-CNN model configuration and weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold for detection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Step 2: Create the predictor
    predictor = DefaultPredictor(cfg)

    # Step 3: Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found. Please check the file path.")
        return

    # Step 4: Run inference on the image
    outputs = predictor(image)

    # Step 5: Visualize the results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Step 6: Display the result
    cv2.imshow("Mask R-CNN Detection", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the Mask R-CNN model on your flag image
image_path = "polandf.png"  # Replace with your image path
run_mask_rcnn(image_path)
