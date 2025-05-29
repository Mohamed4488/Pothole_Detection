from ultralytics import YOLO
from PIL import Image


tuned_model_yolo11n = YOLO(r"runs\detect\train_1\weights\best.pt")
tuned_model_yolo11s = YOLO(r"runs\detect\train_2\weights\best.pt")



def predict_image_n(img_1):
    result_1 = tuned_model_yolo11n.predict(
        img_1
    )
    result_1 = result_1[0]
    
    img_1 = result_1.plot()
    
    pil_image_1 = Image.fromarray(img_1)
    
    return pil_image_1


def predict_image_s(img_2):
    result_2 = tuned_model_yolo11s.predict(
        img_2
    )
    result_2 = result_2[0]
    
    img_2 = result_2.plot()
    
    pil_image_2 = Image.fromarray(img_2)
    
    return pil_image_2
    
if __name__ == "__main__":
    predicted_img_1= predict_image_s(r"potholes-detect\valid\images\1_jpg.rf.6cc98ebd267ba43be7bbef157a3cefc0.jpg")
    predicted_img_1.show()
    
    

