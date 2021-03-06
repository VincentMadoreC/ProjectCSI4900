import correction
import ocr
import cv2
import sys


def main(argv):
    img_path = "./images/cfva648.jpg"
    DEBUG_MODE = False

    for option in argv:
        if option in ('-d', '--debug'):
            print('Running in DEBUG mode...')
            DEBUG_MODE = True
    if len(argv) > 0:
        img_path = argv[0]
    
    # This will only work when correction the code to get the bounding boxes has been completed
    # corrected_img = correction.correct(img_path, True)
    # ocr.ocr(corrected_img, True)


    # In the mean time, correction() and ocr() can be tested individually.
    # Below is sample test code.

    # For now, correction() only works if the bounding boxes have been hardcoded
    correction.correct("images/perspective.jpg", debug_mode=DEBUG_MODE)

    # ocr works with any image
    img = cv2.imread(img_path)
    ocr.ocr(img, debug_mode=DEBUG_MODE)

if __name__ == "__main__":
    main(sys.argv[1:])