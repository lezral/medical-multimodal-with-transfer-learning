from keras.layers import Activation, Input, Dense, Flatten, Dropout, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras import regularizers
from keras.models import Model

__version__ = '0.0.1'


def build_cnn(embedding_layer=None, num_words=None,
              embedding_dim=None, filter_sizes=[3, 4, 5],
              feature_maps=[100, 100, 100], max_seq_length=100, dropout_rate=None, multi=False):

    if len(filter_sizes) != len(feature_maps):
        raise Exception(
            'Please define `filter_sizes` and `feature_maps` with the same length.')
    if not embedding_layer and (not num_words or not embedding_dim):
        raise Exception(
            'Please define `num_words` and `embedding_dim` if you not use a pre-trained embedding')

    print('Creating CNN %s' % __version__)
    print('#############################################')
    print('Embedding:    %s pre-trained embedding' %
          ('using' if embedding_layer else 'no'))
    print('Vocabulary size: %s' % num_words)
    print('Embedding dim: %s' % embedding_dim)
    print('Filter sizes: %s' % filter_sizes)
    print('Feature maps: %s' % feature_maps)
    print('Max sequence: %i' % max_seq_length)
    print('#############################################')

    if embedding_layer is None:
        embedding_layer = Embedding(input_dim=num_words, output_dim=embedding_dim,
                                    input_length=max_seq_length,
                                    weights=None,
                                    trainable=True
                                    )

    channels = []
    x_in = Input(shape=(max_seq_length,), dtype='int32')
    emb_layer = embedding_layer(x_in)
    if dropout_rate:
        emb_layer = Dropout(dropout_rate)(emb_layer)
    for ix in range(len(filter_sizes)):
        x = create_channel(emb_layer, filter_sizes[ix], feature_maps[ix])
        channels.append(x)

    # Concatenate all channels
    x = concatenate(channels)
    concat = concatenate(channels)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    if multi:
        return concat
    return Model(inputs=x_in, outputs=x)


def create_channel(x, filter_size, feature_map):
    """
    Creates a layer working channel wise
    """
    x = Conv1D(feature_map, kernel_size=filter_size, activation='relu', strides=1,
               padding='same', kernel_regularizer=regularizers.l2(0.03))(x)
    x = MaxPooling1D(pool_size=2, strides=1, padding='valid')(x)
    x = Flatten()(x)
    return x
