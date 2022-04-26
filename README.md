# LicensePlateRecognition

## Installation
1. Clone the repository
2. Install Python if you haven't done so already
3. Install OpenCV for python
```
pip install opencv-python
```

## How to use
Run the program using your IDE if you can. There are default values for the image used and the debug mode.
Or use the command line:
```
ProjectCSI4900\LicencePlateRecognition> python main.py ./images/cvfa648.jpg --debug
```
### Manual tuning
In order to properly detect the Alabama plate (bc18351.jpg), search for the comment "for the Alabama plate" then enable the corresponding line, and disable the line below. Also, make sure to provide the bc18351.jpg image.

The image used for correction.py is fixed because the bounding boxes coordinates have to be hardcoded until we get a working localization algorithm.

## Important notes
Not all characters have examples in the standard yet. When running the code with a plate containing numbers that are not in the dataset, make sure to save the characters (by enabling the identified code) and manually add them to the dataset.  Search for the comment “enable this line to create new images to use as standards”.

The program is not really optimized yet. Expect it to take 20-30 seconds to run on slower hardware.

Even though license seems to often be written with a 's', the government of Ontario writes it as licenCe https://www.ontario.ca/page/renew-your-licence-plate :)