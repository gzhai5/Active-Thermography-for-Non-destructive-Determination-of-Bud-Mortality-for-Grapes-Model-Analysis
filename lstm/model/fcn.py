from tensorflow.keras.layers import Input, LSTM, Dropout, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Permute, GlobalAveragePooling1D, concatenate, Dense
from tensorflow.keras.models import Model


def generate_fcn(MAX_SEQUENCE_LENGTH=600, print_summary=False, feature_input_accept=False):
    if not feature_input_accept:
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

        # CNN
        x = Permute((2, 1))(ip)
        conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)

        conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        gap = GlobalAveragePooling1D()(conv3)

        # Classification output
        out = Dense(1, activation='sigmoid')(gap)

        model = Model(ip, out)

        model.summary() if print_summary else None
        return model
    
    else:
        time_series_ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        feature_ip = Input(shape=(7,))

        # CNN
        y = Permute((2, 1))(time_series_ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        # Dense layer for feature input
        z = Dense(64, activation='relu')(feature_ip)
        z = Dropout(0.3)(z)
        z = Dense(32, activation='relu')(z)

        # Merge CNN and LSTM features
        combined = concatenate([y, z])

        # Classification output
        out = Dense(1, activation='sigmoid', name='output')(combined)

        model = Model([time_series_ip, feature_ip], out)

        model.summary() if print_summary else None
        return model