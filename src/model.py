"""Arquitectura CNN (MobileNetV2) + LSTM."""
import tensorflow as tf
from tensorflow.keras import layers, models

import config


def build_model(img_size=config.IMG_SIZE, seq_len=config.SEQ_LEN,
                num_classes=len(config.CLASS_NAMES), pretrained=True,
                lstm_units=128, feature_dim=256):
    """Entrada: (batch, T, H, W, 3) con píxeles en [0, 255]. Salida: softmax de 3 clases."""
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False,
        weights="imagenet" if pretrained else None)
    backbone.trainable = False

    # ---- Codificador espacial (se aplica a cada frame) ----
    frame_in = layers.Input((img_size, img_size, 3), name="frame")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(frame_in)  # → [-1, 1]
    x = backbone(x, training=False)       # BatchNorm SIEMPRE en modo inferencia
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    encoder = models.Model(frame_in, x, name="cnn_encoder")

    # ---- Modelo temporal ----
    seq_in = layers.Input((seq_len, img_size, img_size, 3), name="sequence")
    x = layers.TimeDistributed(encoder, name="time_distributed_cnn")(seq_in)
    x = layers.LSTM(lstm_units, dropout=0.3, name="lstm")(x)  # sin recurrent_dropout → cuDNN
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return models.Model(seq_in, out, name="cnn_lstm_drowsiness")


def get_encoder(model):
    return model.get_layer("time_distributed_cnn").layer


def get_backbone(model):
    for layer in get_encoder(model).layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No se encontró el backbone dentro del codificador.")


def set_backbone_trainable(model, trainable, fine_tune_at=config.FINE_TUNE_AT):
    """Congela/descongela el backbone. En fine-tuning solo se entrenan las capas finales,
    y las BatchNormalization permanecen congeladas (práctica estándar de transfer learning)."""
    backbone = get_backbone(model)
    backbone.trainable = trainable
    if trainable:
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False


def split_encoder_and_head(model):
    """Separa el modelo en (codificador por frame, cabeza temporal sobre features).
    Permite cachear features en inferencia: la CNN procesa cada frame una sola vez."""
    encoder = get_encoder(model)
    names = [l.name for l in model.layers]
    idx = names.index("time_distributed_cnn")
    feature_dim = encoder.output_shape[-1]
    seq_len = model.input_shape[1]
    feat_in = layers.Input((seq_len, feature_dim), name="features")
    x = feat_in
    for layer in model.layers[idx + 1:]:
        x = layer(x)
    return encoder, models.Model(feat_in, x, name="temporal_head")