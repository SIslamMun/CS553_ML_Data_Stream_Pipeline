import cv2
import numpy as np
import base64

def capture_image_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame

def convert_frame_to_base64(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    # print(base64_image)
    return base64_image

def decode_base64_to_frame(base64_image):
    base64_image = base64_image.encode('utf-8')
    image_data = np.frombuffer(base64.b64decode(base64_image), np.uint8)
    frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return frame

# def capture_and_encode_image_from_video(video_path, frame_num):
#     frame = capture_image_from_video(video_path, frame_num)
#     base64_image = convert_frame_to_base64(frame)
#     return base64_image

# def decode_and_display_image(base64_image):
#     frame = decode_base64_to_frame(base64_image)
#     cv2.imshow('Image', frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'single_person.mp4'
    image = capture_image_from_video(video_path)
    base_64 = convert_frame_to_base64(image)
    real_image = decode_base64_to_frame(base_64)
    print(real_image)



