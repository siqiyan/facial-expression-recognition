# import your model library and the open cv library
import YOLO_small_tf_C399, cv2

# put the trained model here
yolo = YOLO_small_tf_C399.YOLO_TF()

# 0 is the 'first' webcam device on this machine
cap = cv2.VideoCapture(0)

# Continuously capture video and detection
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read();

	# Add image processing if needed
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	

	# This model outputs the frame with bounding box
	yolo.detect_from_cvmat(frame);

    # Display the resulting frame with OpenCV

    #cv2.imshow('frame',gray)
    
    # Detect 'q' for quit
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break


# When everything done, release the capture
# Note, we usually don't get this far.
cap.release()
cv2.destroyAllWindows()
