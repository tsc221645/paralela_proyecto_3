"""Arquitecturas de modelos para clasificación ASL."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_data_augmentation() -> keras.Sequential:
    """Aumento de datos moderado; sin flip horizontal (cambia el significado ASL)."""
    return keras.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
            layers.RandomZoom(height_factor=(-0.05, 0.10), width_factor=(-0.05, 0.10)),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def build_dense_model(
    input_shape: tuple[int, int, int],
    number_of_classes: int,
) -> keras.Model:
    """Red fully-connected como línea base."""
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.40),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.30),
            layers.Dense(number_of_classes, activation="softmax"),
        ],
        name="dense_baseline",
    )


def build_basic_cnn(
    input_shape: tuple[int, int, int],
    number_of_classes: int,
) -> keras.Model:
    """CNN básica para patrones espaciales simples."""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(number_of_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_basica")


def build_deep_cnn(
    input_shape: tuple[int, int, int],
    number_of_classes: int,
    dropout_rate: float = 0.40,
    use_augmentation: bool = False,
) -> keras.Model:
    """CNN más profunda. El baseline no usa augmentation para comparar de forma justa."""
    inputs = keras.Input(shape=input_shape)
    x = inputs

    if use_augmentation:
        x = build_data_augmentation()(x)

    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(number_of_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_profunda")


def compile_model(model: keras.Model, learning_rate: float = 0.001) -> None:
    """Compilación prevista para la entrega final."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
