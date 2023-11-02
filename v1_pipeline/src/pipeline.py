import os
from src.person import person_detector

class Non_Productive_Time:

    def __init__(self):
        #initializing directory path
        self.dir_path = os.path.dirname(os.path.dirname(__file__))
        self.absent_time_calc = {}
        #initializing absent time calculation dictionary
        self.inactivity_time_calc = {}


    # person  output pipeline
    def pipeline(self, image, date_time,
    image_shape,person_model,qat_pep,pfp16,person_has_postprocessing,p_conf_thresh,engine,device):
        """
        DETECTION PIPELINE
        Args:
            camera_id: Camera id of the image.
            roi_boxs: List of ROI boxes.
            image_path: Path to the image to detect objects in.
            frame_id: Frame id of the image.
            date_time: Date and time of the image.

        Returns: 
            present_absent: Present or absent status of the person.

        """
        cropped_input_img = image

        print("i am in pipeline")

        #calling person detector
        person_count,_ = person_detector(cropped_input_img,image_shape,
        person_model,qat_pep,pfp16,person_has_postprocessing,p_conf_thresh,engine,device)
        time_calc = {"value":int(person_count),"time":date_time,"topic":"absentPresent"}
        print(time_calc)

        return time_calc
