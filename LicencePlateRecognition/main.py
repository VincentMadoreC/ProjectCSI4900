import correction
import ocr
import cv2


if __name__ == "__main__":
    # img_path = "./images/bc18351.jpg"
    img_path = "./images/cfva648.jpg"

    # This will only work when correction the code to get the bounding boxes has been completed
    # corrected_img = correction.correct(img_path, True)
    # ocr.ocr(corrected_img, True)


    # In the mean time, correction() and ocr() can be tested individually.
    # Below is sample test code.

    # For now, correction() only works if the bounding boxes have been hardcoded
    correction.correct("images/perspective.jpg", debug_mode=True)

    # ocr works with any image
    img = cv2.imread(img_path)
    ocr.ocr(img, debug_mode=True)