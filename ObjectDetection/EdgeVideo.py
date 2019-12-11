from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import time
import cv2

labelfile = '../TFLiteModel/mobilenet_ssd_v2/coco_labels.txt'
modelfile = '../TFLiteModel/mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'


CONFIDENCE = 0.3

# initialize the labels dictionary
print("Parsing class labels...")
labels = {}

# loop over the class labels file
for row in open(labelfile):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()

# load the Google Coral object detection model
print("Loading Coral model...")
model = DetectionEngine(modelfile)

# initialize the video stream and allow the camera sensor to warmup
print("Starting video stream...")
video = '../TestVid/shazam.mp4'

cap = cv2.VideoCapture(video)
time.sleep(2.0)
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)
        orginal = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        results = model.DetectWithImage(frame, threshold=CONFIDENCE
                                        ,keep_aspect_ratio=True,
                                        relative_coord=False)

        # loop over the results
        for r in results:
            # extract the bounding box and box and predicted class label
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box
            label = labels[r.label_id]

            # draw the bounding box and label on the image
            cv2.rectangle(orginal, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv2.putText(orginal, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the output frame and wait for a key press
        cv2.imshow("Frame", orginal)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()




