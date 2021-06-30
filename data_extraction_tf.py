import main
from globals import *

def data_extraction():

    IMG_PATH = train_data + 'images/t000.tif'
    MASK_PATH = train_data + 'masks/SEG/man_seg000.tif'

    print("Information about data: ")

    TRAINSET_SIZE = len(glob(train_data + 'images/*.tif'))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

    TESTSET_SIZE = len(glob(test_data + 'images/*.tif'))
    print(f"The Test Dataset contains {TESTSET_SIZE} images.")

    sample_image = Image.open(IMG_PATH)
    input_size = sample_image.size
    input_width = input_size[0]
    input_height = input_size[1]
    print("Image size: ", input_size)
    print("Image format: ", sample_image.format)
    print(f"Image mode: {sample_image.mode} (16-bit unsigned integer pixels)")
    print()

    sample_mask = Image.open(MASK_PATH)
    mask_size = sample_mask.size
    print("Mask size: ", mask_size)
    print("Mask format: ", sample_mask.format)
    print(f"Mask mode: {sample_mask.mode} (16-bit unsigned integer pixels)")
    print()

    sample_image = train_data + 'images/t000.tif'
    image = imread(sample_image)    # read the image using skimage
    plt.imshow(image)
    plt.waitforbuttonpress()

    sample_mask = train_data + 'masks/SEG/man_seg000.tif'
    mask = imread(sample_mask)
    plt.imshow(mask)
    plt.waitforbuttonpress()

    for i, image_id in tqdm(enumerate(train_ids), total=len(train_ids)):
        path_image = train_data + 'images/' + image_id

        # read the image using skimage
        image = imread(path_image)

        # resize the image
        image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        image = np.expand_dims(image, axis=-1)

        # insert the image into X_train
        X_train[i] = image

    print(f"X_train shape: {X_train.shape}")

    # Y_train

    for i, mask_id in tqdm(enumerate(train_mask_ids), total=len(train_ids)):
        path_mask = train_data + 'masks/SEG/' + mask_id

        # read the image using skimage
        mask = imread(path_mask)

        # resize the image
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        mask = np.expand_dims(mask, axis=-1)

        # insert the image into Y_Train
        Y_train[i] = mask

    print(f"y_train shape: {Y_train.shape}")

    # X_test

    for i, test_id in tqdm(enumerate(test_ids), total=len(test_ids)):
        path_image = test_data + 'images/' + test_id

        # read the image using skimage
        img = imread(path_image)

        # resize the image
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        img = np.expand_dims(img, axis=-1)

        # insert the image into Y_Train
        X_test[i] = img

    print(f"X_test shape: {X_test.shape}")

    # Y_test

    for i, mask_id in tqdm(enumerate(test_mask_ids), total=len(test_ids)):
        path_mask = test_data + 'masks/SEG/' + mask_id

        # read the image using skimage
        mask = imread(path_mask)

        # resize the image
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

        # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
        mask = np.expand_dims(mask, axis=-1)

        # insert the image into Y_Train
        Y_test[i] = mask

    print(f"Y_test shape: {Y_test.shape}")

    print("... Data extracted")


    """
    # conversion of tif to png
    print("... converting tif to png")
    for filename in glob(train_data+'images/*.tif'):
        temp = Image.open(filename)
        temp_rgb = temp.convert('RGB')
        temp_rgb.save(filename.replace("tif", "png"))

    for filename in glob(train_data+'masks/SEG/*.tif'):
        temp = Image.open(filename)
        temp_rgb = temp.convert('RGB')
        temp_rgb.save(filename.replace("tif", "png"))

    for filename in glob(val_data+'images/*.tif'):
        temp = Image.open(filename)
        temp_rgb = temp.convert('RGB')
        temp_rgb.save(filename.replace("tif", "png"))

    for filename in glob(val_data+'masks/SEG/*.tif'):
        temp = Image.open(filename)
        temp_rgb = temp.convert('RGB')
        temp_rgb.save(filename.replace("tif", "png"))

    print("... done with the conversion")

    print("... resizing the images to 600x600")
    # resize images to 600x600
    # https://auth0.com/blog/image-processing-in-python-with-pillow/
    # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

    #train images
    for filename,i in zip(glob(train_data + 'images/*.png'),range(150)):
        sample_image = Image.open(filename)
        new_image = sample_image.resize((600,600), resample=Image.NEAREST)
        new_image.save(train700_data+'images/'+str(i)+'.png')

    # train masks
    new_mask_path = train700_data+'masks/'
    i = 0
    for filename in sorted(glob(train_data + 'masks/SEG/*.png')):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        dim = (600, 600)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(new_mask_path+str(i)+'.png',resized)
        i += 1

    # val images
    for filename,i in zip(glob(val_data + 'images/*.png'),range(65)):
        sample_image = Image.open(filename)
        new_image = sample_image.resize((600,600), resample=Image.NEAREST)
        new_image.save(val700_data+'images/'+str(i)+'.png')

    # val masks
    for filename,i in zip(sorted(glob(val_data + 'masks/SEG/*.png')),range(65)):
        sample_mask = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        new_mask = cv2.resize(sample_mask, (600,600), interpolation = cv2.INTER_AREA)
        cv2.imwrite(val700_data+'masks/'+str(i)+'.png',new_mask)

    print("... done resizing")
    """

    # # original image
    # img = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
    # window_name = 'img'
    # cv2.imshow(window_name, img)
    # cv2.waitKey(0)
    #
    # # resized image
    # img = cv2.imread(train700_data + 'images/0.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow(window_name, img)
    # cv2.waitKey(0)
    #
    # # original mask
    # mask = cv2.imread(train_data + 'masks/SEG/man_seg000.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow(window_name, mask * 10)
    # cv2.waitKey(0)
    #
    # # resized mask
    # mask = cv2.imread(train700_data + 'masks/' + '0.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow(window_name, mask * 10)
    # cv2.waitKey(0)

