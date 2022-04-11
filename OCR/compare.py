def compare_images(image1, image2):
    '''
    Inspired by this https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    Use the Mean Squared Error (MSE) to compare the two digits. 
    The two images must be the same size, so we match their height while keeping the aspect ratio,
    then we stretch the most narrow one to match their width without changing the height. 
    Both images need to have the same dimensions to be compared.
        It is okay to stretch the image without keeping the aspect ratio, 
        because two equal digits will have roughly the same shape, so they won’t be stretched too much, 
        while different digits will be stretched more, giving a worse (higher) MSE value. 
        This makes it less likely to get a match where we don’t want to.
    '''

    # Isolate the number part of each image
    img_isolated1 = isolate_image(np.array(image1))
    img_isolated2 = isolate_image(np.array(image2))

    # Resize the smallest one to match the dimensions of the biggest one
    max_height = max(img_isolated1.shape[0], img_isolated2.shape[0])
    # Adjust the height of the smallest image to match the height of the biggest, while keeping proportions
    if (img_isolated1.shape[0] < img_isolated2.shape[0]):
        ratio = max_height/img_isolated1.shape[0]
        img_isolated1 = cv2.resize(img_isolated1, (0, 0), fx=ratio, fy=ratio)
    else:
        ratio = max_height/img_isolated2.shape[0]
        img_isolated2 = cv2.resize(img_isolated2, (0, 0), fx=ratio, fy=ratio)

    # Adjust the width of the most narrow image to match the width of the widest, without changing the height
    max_width = max(img_isolated1.shape[1], img_isolated2.shape[1])
    if (img_isolated1.shape[1] < img_isolated2.shape[1]):
        ratio = max_width/img_isolated1.shape[1]
        img_isolated1 = cv2.resize(img_isolated1, (0, 0), fx=ratio, fy=1)
    else:
        ratio = max_width/img_isolated2.shape[1]
        img_isolated2 = cv2.resize(img_isolated2, (0, 0), fx=ratio, fy=1)

    # Calculate the Mean Squared Error
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    img_isolated1 = img_isolated1/255
    img_isolated2 = img_isolated2/255
    mse = np.sum((img_isolated1 - img_isolated2) ** 2)
    mse /= float(img_isolated1.shape[0] * img_isolated1.shape[1])

    return mse


def compare_against_dataset(image):
    '''
    Compare the specified image against the dataset.
    Store the average MSE for each number.
    Check which number has the best (lowest) MSE on average when compared to the image.
    '''
    mse_list = []
    path_dataset = IMAGE_FOLDER + "/standard"
    for folder in os.listdir(path_dataset):
        # Skip if it's just an unclassified image
        if folder.endswith(".png") or folder.endswith(".jpg"):
            continue
        path_folder = path_dataset + "/" + folder
        count = 0
        sum_mse = 0
        for example in os.listdir(path_folder):
            path_example = path_folder + "/" + example
            standard = cv2.imread(path_example, cv2.IMREAD_GRAYSCALE)
            sum_mse += compare_images(standard, image)
            count += 1
        mse_list.append(sum_mse/count)

    best_mse = 1
    best_mse_index = 0
    # Find the best (lowest) MSE and the corresponding folder
    for i, mse in enumerate(mse_list):
        if mse < best_mse:
            best_mse = mse
            best_mse_index = i
    return best_mse_index