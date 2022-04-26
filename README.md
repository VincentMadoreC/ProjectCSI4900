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

## Important notes
Not all characters have examples in the standard yet. When running the code with a plate containing numbers that are not in the dataset, make sure to save the characters (by enabling the identified code) and manually add them to the dataset.  Search for the comment “enable this line to create new images to use as standard”.

The program is not really optimized yet. Expect it to take 20-30 seconds to run on slower hardware.

Even though license seems to often be written with a 's', the government of Ontario writes it as licenCe https://www.ontario.ca/page/renew-your-licence-plate :)