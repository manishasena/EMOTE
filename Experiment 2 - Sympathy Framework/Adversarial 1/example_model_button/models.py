import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
from keras.layers import Input, Dense,Flatten, Reshape, Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, MaxPool2D, Lambda, Activation, Subtract
from keras.backend import abs, mean
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.activations import softmax, sigmoid
from keras.metrics import binary_crossentropy, categorical_accuracy
from keras.utils import plot_model
#from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
#from tensorflow.python.keras.layers.core import Activation

#seed = 1
#import tensorflow as tf
#np.random.seed(seed)
#np.random.seed(seed)
#tf.random.set_seed(seed)

import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorflow as tf
#from keras import backend as K
from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

# Q_{symp}
class DQN:

    def __init__(self, width, height, num_actions, params):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = Conv2D(filters=32, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        self.outputs = Dense(self.num_actions)(x)
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='mse',metrics=['accuracy'])

        if params['load_file'] is not None:
            print('Loading checkpoint...')

            model_location = params['save_dir'] + '/'
            save_location = model_location + 'GridWorld_LearningAgent_' + params['load_file']
            self.model = load_model(save_location, custom_objects={'tf': tf}, compile = False)

# Selfish DQN
class DQN_Greedy:

    def __init__(self, width, height, num_actions):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = Conv2D(filters=32, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        self.outputs = Dense(self.num_actions)(x)
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='mse',metrics=['accuracy'])

# Empathetic Independent Agent model (FEATURE) - Modelled Button Status
class DQN_Human_pixel:

    def __init__(self, width, height, num_actions, params):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions
        loss1 = params['loss1_weight']

        ##############################################
        # Pixle manipulation network (same weight transformation to each pixel)

        self.pixle_input = Input(shape=(1,))
        no_levels = 6

        x = self.pixle_input
        x = Dense(no_levels, activation = "relu")(x)
        x = Dense(no_levels**2, activation = "relu")(x)
        x = Dense(no_levels**2, activation = "sigmoid")(x)
        self.pixel_output = Dense(1, activation = "sigmoid", name = "pixel_output")(x)
        self.state_model = Model(self.pixle_input, self.pixel_output, name = 'single_pixel_net')

        ##############################################
        # Power pellet
        self.power_in = Input(shape = (1,))

        x = self.power_in
        x = Lambda(lambda x: x * 2 - 1)(x)
        x = Lambda(lambda x: x * 6)(x)
        self.power_out = Dense(1, activation="sigmoid", use_bias=False)(x)
        self.power_model = Model(self.power_in, self.power_out, name = 'power_model')
        optimiser = Adam(lr = 0.01)
        self.power_model.compile(optimizer=optimiser, loss= 'mse')

        ##############################################
        # DQN Selfish

        self.inputs2 = Input(shape=(width,height,1,))
        self.door2 = Input(shape=(1,))
        self.R_x2 = Input(shape=(1,))
        self.R_y2 = Input(shape=(1,))
        self.H_x2 = Input(shape=(1,))
        self.H_y2 = Input(shape=(1,))

        x = self.inputs2
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = Conv2D(filters=32, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door2, self.R_x2, self.R_y2, self.H_x2, self.H_y2])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        
        self.bsm = Dense(self.num_actions)(x)
        self.Greedy_model = Model([self.inputs2, self.door2, self.R_x2, self.R_y2, self.H_x2, self.H_y2], self.bsm, name='q_net_greedy_model')
        self.Greedy_model.trainable = False

        ##############################################

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        # Model vision
        self.pout = self.state_model(self.inputs)
        # Model button status
        self.door_t = self.power_model(self.door)

        # Feed into Selfish network
        decoded_img = self.Greedy_model([self.pout, self.door_t, self.H_x, self.H_y, self.R_x, self.R_y])  #SWAPPING LOCATIONS!
        self.model_bsm = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], decoded_img, name="q_net_bsm")

        # After softmax
        self.outputs = Activation(softmax, name = "final_softmax")(decoded_img)
        ###
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net_human')

        #Compile
        optimiser = Adam(lr = 0.001)

        # Calculate custom loss (image transformation)
        image_loss = 4*loss1*mean(abs(Subtract()([self.inputs, self.pout])))

        # Compile
        self.model.add_loss(image_loss)

        def _to_tensor(x, dtype):
            """Convert the input `x` to a tensor of type `dtype`.
            # Arguments
                x: An object to be converted (numpy array, list, tensors).
                dtype: The destination type.
            # Returns
                A tensor.
            """
            x = tf.convert_to_tensor(x)
            if x.dtype != dtype:
                x = tf.cast(x, dtype)
            return x

        # Keras categorical cross entropy code:
        def categorical_crossentropy_code(y_true, y_pred, from_logits=False):
            """Categorical crossentropy between an y_pred tensor and a y_true tensor.
            # Arguments
                y_pred: A tensor resulting from a softmax
                    (unless `from_logits` is True, in which
                    case `y_pred` is expected to be the logits).
                y_true: A tensor of the same shape as `y_pred`.
                from_logits: Boolean, whether `y_pred` is the
                    result of a softmax, or is a tensor of logits.
            # Returns
                y_pred tensor.
            """
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
            if not from_logits:

                # scale preds so that the class probas of each sample sum to 1
                y_pred /= tf.reduce_sum(y_pred,
                                        axis=len(y_pred.get_shape()) - 1,
                                        keepdims=True)
                # manual computation of crossentropy
                _EPSILON = 1e-07
                epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

                loss2_value = - tf.reduce_sum(y_true * tf.math.log(y_pred),
                                    axis=len(y_pred.get_shape()) - 1)
                return loss2_value*1*(1-loss1)
            else:
                return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)

        self.model.compile(optimizer=optimiser, loss=categorical_crossentropy_code, metrics=['categorical_accuracy'])

        if params['load_file'] is not None:
            print('Loading checkpoint...')

            model_location = 'example_' + str(params['run_no']) + '/saves/' + params['save_val'] + '_pixel_' + str(int(params['loss1_weight']*100)) + '/'
            save_location = model_location + 'GridWorld_StateTransform_' + params['load_file']
            self.state_model = load_model(save_location, custom_objects={'tf': tf}, compile = False)

            model_location = 'example_' + str(params['run_no']) + '/saves/' + params['save_val'] + '_pixel_' + str(int(params['loss1_weight']*100)) + '/'
            save_location = model_location + 'GridWorld_ButtonModel_' + params['load_file']
            self.power_model = load_model(save_location, custom_objects={'tf': tf}, compile = False)

# Empathetic Independent Agent model (IMAGE) - Modelled Button Status
class DQN_Human_state:

    def __init__(self, width, height, num_actions, params):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions
        loss1 = params['loss1_weight']

        ###########################

        self.state_input = Input(shape=(width,height,1,))
        self.state_out_shape = width*height
        no_levels = 6

        x = self.state_input
        x = Conv2D(no_levels**2, kernel_size = (2,2), activation = "relu")(x)
        x = Conv2D(no_levels, kernel_size = (2,2), activation = "relu")(x)
        x = Conv2DTranspose(no_levels, kernel_size = (2,2), activation = "relu")(x)
        x = Conv2DTranspose(no_levels, kernel_size = (2,2), activation = "relu")(x)
        x = Conv2D(1, kernel_size = (1,1), activation = "sigmoid")(x)
        
        self.state_output = x
        self.state_model = Model(self.state_input, self.state_output, name = 'state_net')
        
        ##############################################
        # Power pellet
        self.power_in = Input(shape = (1,))

        x = self.power_in
        x = Lambda(lambda x: x * 2 - 1)(x)
        x = Lambda(lambda x: x * 6)(x)
        self.power_out = Dense(1, activation="sigmoid", use_bias=False)(x)
        self.power_model = Model(self.power_in, self.power_out, name = 'power_model')
        optimiser = Adam(lr = 0.01)
        self.power_model.compile(optimizer=optimiser, loss= 'mse')

        ##############################################
        # DQN Selfish

        self.inputs2 = Input(shape=(width,height,1,))
        self.door2 = Input(shape=(1,))
        self.R_x2 = Input(shape=(1,))
        self.R_y2 = Input(shape=(1,))
        self.H_x2 = Input(shape=(1,))
        self.H_y2 = Input(shape=(1,))

        x = self.inputs2
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = Conv2D(filters=32, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door2, self.R_x2, self.R_y2, self.H_x2, self.H_y2])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)

        self.bsm = Dense(self.num_actions)(x)
        self.Greedy_model = Model([self.inputs2, self.door2, self.R_x2, self.R_y2, self.H_x2, self.H_y2], self.bsm, name='q_net_greedy_model')
        self.Greedy_model.trainable = False

        ##############################################

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        # Apply transformation to state 
        self.pout = self.state_model(self.inputs)
        # Model Button
        self.door_t = self.power_model(self.door) 
        
        # Feed into selfish model
        decoded_img = self.Greedy_model([self.pout, self.door_t, self.H_x, self.H_y, self.R_x, self.R_y])  #SWAPPING LOCATIONS!
        self.model_bsm = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], decoded_img, name="q_net_bsm")

        # After softmax
        self.outputs = Activation(softmax, name = "final_softmax")(decoded_img)
        # Compile model
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net_human')

        #Compile
        optimiser = Adam(lr = 0.0001)

        # Calculate custom loss (image transformation)
        image_loss = loss1*8*mean(abs(Subtract()([self.inputs, self.pout])))

        # Compile
        self.model.add_loss(image_loss)

        def _to_tensor(x, dtype):
            """Convert the input `x` to a tensor of type `dtype`.
            # Arguments
                x: An object to be converted (numpy array, list, tensors).
                dtype: The destination type.
            # Returns
                A tensor.
            """
            x = tf.convert_to_tensor(x)
            if x.dtype != dtype:
                x = tf.cast(x, dtype)
            return x

        # Keras categorical cross entropy code:
        def categorical_crossentropy_code(y_true, y_pred, from_logits=False):
            """Categorical crossentropy between an y_pred tensor and a y_true tensor.
            # Arguments
                y_pred: A tensor resulting from a softmax
                    (unless `from_logits` is True, in which
                    case `y_pred` is expected to be the logits).
                y_true: A tensor of the same shape as `y_pred`.
                from_logits: Boolean, whether `y_pred` is the
                    result of a softmax, or is a tensor of logits.
            # Returns
                y_pred tensor.
            """
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
            if not from_logits:

                # scale preds so that the class probas of each sample sum to 1
                y_pred /= tf.reduce_sum(y_pred,
                                        axis=len(y_pred.get_shape()) - 1,
                                        keepdims=True)
                # manual computation of crossentropy
                _EPSILON = 1e-07
                epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

                loss2_value = - tf.reduce_sum(y_true * tf.math.log(y_pred),
                                    axis=len(y_pred.get_shape()) - 1)
                return loss2_value*1*(1-loss1)
            else:
                return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)

        self.model.compile(optimizer=optimiser, loss=categorical_crossentropy_code, metrics=['categorical_accuracy'])

        if params['load_file'] is not None:
            print('Loading checkpoint...')

            model_location = 'example_' + str(params['run_no']) + '/saves/' + params['save_val'] + '_state_' + str(int(params['loss1_weight']*100)) + '/'
            save_location = model_location + 'GridWorld_StateTransform_' + params['load_file']
            self.state_model = load_model(save_location, custom_objects={'tf': tf}, compile = False)

            model_location = 'example_' + str(params['run_no']) + '/saves/' + params['save_val'] + '_state_' + str(int(params['loss1_weight']*100)) + '/'
            save_location = model_location + 'GridWorld_ButtonModel_' + params['load_file']
            self.power_model = load_model(save_location, custom_objects={'tf': tf}, compile = False)

# Sympathy
class DQN_Human_sympathetic:

    def __init__(self, width, height, num_actions):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = Conv2D(filters=32, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)

        ### before soft max
        self.bsm = Dense(self.num_actions)(x)
        self.model_bsm = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.bsm, name='q_net_bsm')
        self.outputs = Activation(softmax)(self.bsm)
        ###
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net_human')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])

# Benchmark
class DQN_Human_benchmark:

    def __init__(self, width, height, num_actions, params):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.num_actions = num_actions
        loss1 = params['loss1_weight']

        ##############################################
        # DQN Greedy

        self.inputs2 = Input(shape=(width,height,1,))
        self.door2 = Input(shape=(1,))
        self.R_x2 = Input(shape=(1,))
        self.R_y2 = Input(shape=(1,))
        self.H_x2 = Input(shape=(1,))
        self.H_y2 = Input(shape=(1,))

        x = self.inputs2
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = Conv2D(filters=32, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door2, self.R_x2, self.R_y2, self.H_x2, self.H_y2])
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)

        self.bsm = Dense(self.num_actions)(x)
        self.Greedy_model = Model([self.inputs2, self.door2, self.R_x2, self.R_y2, self.H_x2, self.H_y2], self.bsm, name='q_net_greedy_model')
        self.Greedy_model.trainable = False

        ##############################################

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        # Apply transformation to state and door
        self.pout = self.inputs
        self.door_t = self.door

        decoded_img = self.Greedy_model([self.pout, self.door_t, self.H_x, self.H_y, self.R_x, self.R_y])  #SWAPPING LOCATIONS!
        self.model_bsm = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], decoded_img, name="q_net_bsm")

        # After softmax
        self.outputs = Activation(softmax, name = "final_softmax")(decoded_img)
        # Compile model
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y], self.outputs, name='q_net_human')

        #Compile
        optimiser = Adam(lr = 0.0001)

        # Calculate custom loss (image transformation)
        image_loss = loss1*4*mean(abs(Subtract()([self.inputs, self.pout])))

        # Compile
        self.model.add_loss(image_loss)

        def _to_tensor(x, dtype):
            """Convert the input `x` to a tensor of type `dtype`.
            # Arguments
                x: An object to be converted (numpy array, list, tensors).
                dtype: The destination type.
            # Returns
                A tensor.
            """
            x = tf.convert_to_tensor(x)
            if x.dtype != dtype:
                x = tf.cast(x, dtype)
            return x

        # Keras categorical cross entropy code:
        def categorical_crossentropy_code(y_true, y_pred, from_logits=False):
            """Categorical crossentropy between an y_pred tensor and a y_true tensor.
            # Arguments
                y_pred: A tensor resulting from a softmax
                    (unless `from_logits` is True, in which
                    case `y_pred` is expected to be the logits).
                y_true: A tensor of the same shape as `y_pred`.
                from_logits: Boolean, whether `y_pred` is the
                    result of a softmax, or is a tensor of logits.
            # Returns
                y_pred tensor.
            """
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
            if not from_logits:

                # scale preds so that the class probas of each sample sum to 1
                y_pred /= tf.reduce_sum(y_pred,
                                        axis=len(y_pred.get_shape()) - 1,
                                        keepdims=True)
                # manual computation of crossentropy
                _EPSILON = 1e-07
                epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
                y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

                loss2_value = - tf.reduce_sum(y_true * tf.math.log(y_pred),
                                    axis=len(y_pred.get_shape()) - 1)
                return loss2_value*0.25*(1-loss1)
            else:
                return tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)

        #self.model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.compile(optimizer=optimiser, loss=categorical_crossentropy_code, metrics=['categorical_accuracy'])

# Component of Sympathy Framework Next State Model
class nextStateModel:

    def __init__(self, width, height):

        # Specifying number of actions - left, right, forward, pick up, toggle
        self.out_shape = width*height + 5

        self.inputs = Input(shape=(width,height,1,))
        self.door = Input(shape=(1,))
        self.R_x = Input(shape=(1,))
        self.R_y = Input(shape=(1,))
        self.H_x = Input(shape=(1,))
        self.H_y = Input(shape=(1,))

        self.action = Input(shape=(1,))

        x = self.inputs
        x = Conv2D(filters=16, kernel_size = (2,2), padding = "same", activation="relu")(x)
        x = MaxPool2D(pool_size=(2,2), strides = (1,1))(x)
        x = Flatten()(x)

        x = Concatenate(axis = 1)([x, self.door, self.R_x, self.R_y, self.H_x, self.H_y])
        x = Concatenate(axis = 1)([x, self.action])
        x = Dense(32, activation="relu")(x)
        x = Dense(16, activation="relu")(x)

        ### before soft max
        x = Dense(self.out_shape)(x)
        self.outputs = Activation(sigmoid)(x)
        self.model = Model([self.inputs, self.door, self.R_x, self.R_y, self.H_x, self.H_y, self.action], self.outputs, name='nextmodel')

        #Compile
        optimiser = Adam(lr = 0.0001)

        self.model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])
