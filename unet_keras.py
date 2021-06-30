from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

import main
# from unet_utils import *
from globals import *



# def UNET(a, b, c, d, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS):
#     # Build U-Net model
#     inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#     s = Lambda(lambda x: x / 255) (inputs)
#     c1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
#     c1 = Dropout(0.1) (c1)
#     c1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
#     p1 = MaxPooling2D((2, 2)) (c1)
#     c2 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
#     c2 = Dropout(0.1) (c2)
#     c2 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
#     p2 = MaxPooling2D((2, 2)) (c2)
#     c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
#     c3 = Dropout(0.2) (c3)
#     c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
#     p3 = MaxPooling2D((2, 2)) (c3)
#     c4 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
#     c4 = Dropout(0.2) (c4)
#     c4 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
#     p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#     c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
#     c5 = Dropout(0.3) (c5)
#     c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
#     u6 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same') (c5)
#     u6 = concatenate([u6, c4])
#     c6 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
#     c6 = Dropout(0.2) (c6)
#     c6 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
#     u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
#     u7 = concatenate([u7, c3])
#     c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
#     c7 = Dropout(0.2) (c7)
#     c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
#     u8 = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same') (c7)
#     u8 = concatenate([u8, c2])
#     c8 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
#     c8 = Dropout(0.1) (c8)
#     c8 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
#     u9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c8)
#     u9 = concatenate([u9, c1], axis=3)
#     c9 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
#     c9 = Dropout(0.1) (c9)
#     c9 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
#     outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
#     model = Model(inputs=[inputs], outputs=[outputs])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou(), 'accuracy'])
#
#     #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#     model.summary()
#
#     # Fit model
#     earlystopper = EarlyStopping(patience=5, verbose=1)
#     checkpointer = ModelCheckpoint('model-dsbowl2018-2.h5', verbose=1, save_best_only=True)
#     results = model.fit(a,b,batch_size=256,verbose=1,epochs=40,validation_data=(c,d),callbacks = [earlystopper, checkpointer, MetricsCheckpoint('logs')])
#
#     plot_learning_curve(results)
#     plt.show()
#
#     plotKerasLearningCurve()
#     plt.show()
#     global modelZ
#     modelZ = model
#
#     return modelZ

# source: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

def UNET():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(1024, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.summary()

    filepath = "model.h5"

    earlystopper = EarlyStopping(patience=5, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    callbacks_list = [earlystopper, checkpoint]

    history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=5,
                        callbacks=callbacks_list)

    model.save(dataset_path+'saved_model/my_model')

    print("... UNet trained")


def result_inspection():
    model = tf.keras.models.load_model(dataset_path+'saved_model/my_model')

    # Make a prediction
    # use the best epoch
    model.load_weights('model.h5')

    test_preds = model.predict(X_test)

    # Threshold the predictions

    preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)

    # Display a threshold mask

    test_img = preds_test_thresh[5, :, :, 0]

    plt.imshow(test_img, cmap='gray')
    plt.waitforbuttonpress()

    # # set up the canvas for the subplots
    # plt.figure(figsize=(10, 10))
    # plt.axis('Off')
    #
    # # Our subplot will contain 3 rows and 3 columns
    # # plt.subplot(nrows, ncols, plot_number)
    #
    # # == row 1 ==
    #
    # # image
    # plt.subplot(3, 3, 1)
    # test_image = X_test[1, :, :, 0]
    # plt.imshow(test_image)
    # plt.title('Test Image', fontsize=14)
    # plt.axis('off')
    #
    # # true mask
    # plt.subplot(3, 3, 2)
    # mask_id = train_mask_ids
    # path_mask = train_data + 'masks/SEG/' + mask_id
    # mask = imread(path_mask)
    # mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # plt.imshow(mask, cmap='gray')
    # plt.title('True Mask', fontsize=14)
    # plt.axis('off')
    #
    # # predicted mask
    # plt.subplot(3, 3, 3)
    # test_mask = preds_test_thresh[1, :, :, 0]
    # plt.imshow(test_mask, cmap='gray')
    # plt.title('Pred Mask', fontsize=14)
    # plt.axis('off')
    #
    # # == row 2 ==
    #
    # # image
    # plt.subplot(3, 3, 4)
    # test_image = X_test[2, :, :, 0]
    # plt.imshow(test_image)
    # plt.title('Test Image', fontsize=14)
    # plt.axis('off')
    #
    # # true mask
    # plt.subplot(3, 3, 5)
    # mask_id = df_test.loc[2, 'mask_id']
    # path_mask = train_ + mask_id
    # mask = imread(path_mask)
    # mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # plt.imshow(mask, cmap='gray')
    # plt.title('True Mask', fontsize=14)
    # plt.axis('off')
    #
    # # predicted mask
    # plt.subplot(3, 3, 6)
    # test_mask = preds_test_thresh[2, :, :, 0]
    # plt.imshow(test_mask, cmap='gray')
    # plt.title('Pred Mask', fontsize=14)
    # plt.axis('off')
    #
    # # == row 3 ==
    #
    # # image
    # plt.subplot(3, 3, 7)
    # test_image = X_test[3, :, :, 0]
    # plt.imshow(test_image)
    # plt.title('Test Image', fontsize=14)
    # plt.axis('off')
    #
    # # true mask
    # plt.subplot(3, 3, 8)
    # mask_id = df_test.loc[3, 'mask_id']
    # path_mask = '../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id
    # mask = imread(path_mask)
    # mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # plt.imshow(mask, cmap='gray')
    # plt.title('True Mask', fontsize=14)
    # plt.axis('off')
    #
    # # predicted mask
    # plt.subplot(3, 3, 9)
    # test_mask = preds_test_thresh[3, :, :, 0]
    # plt.imshow(test_mask, cmap='gray')
    # plt.title('Pred Mask', fontsize=14)
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    #
    print("... Prediction is over")
