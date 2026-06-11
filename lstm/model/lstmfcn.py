from tensorflow.keras.layers import Input, LSTM, Dropout, Conv1D, BatchNormalization, Activation
from tensorflow.keras.layers import Permute, GlobalAveragePooling1D, concatenate, Dense
from tensorflow.keras.models import Model


def generate_lstmfcn(MAX_SEQUENCE_LENGTH=600, NUM_CELLS=8, dropout=0.8, print_summary=False, feature_input_accept=False):
    if not feature_input_accept:
        ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

        # LSTM
        x = LSTM(NUM_CELLS)(ip)
        x = Dropout(dropout)(x)

        # CNN
        y = Permute((2, 1))(ip)
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

        # Merge CNN and LSTM features
        x = concatenate([x, y])

        # Classification output
        out = Dense(1, activation='sigmoid')(x)

        model = Model(ip, out)

        model.summary() if print_summary else None
        return model
    else:
        time_series_ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))
        feature_ip = Input(shape=(7,))

        # LSTM
        x = LSTM(NUM_CELLS)(time_series_ip)
        x = Dropout(dropout)(x)

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
        combined = concatenate([x, y, z])

        # Classification output
        out = Dense(1, activation='sigmoid', name='output')(combined)

        model = Model([time_series_ip, feature_ip], out)

        model.summary() if print_summary else None
        return model