from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """
    Builds a professional-grade Diagnostic Model using MobileNetV2.
    Optimized for medical image feature extraction.
    """
    print("[INFO] 🏗️ Building Medical Diagnostic Architecture...")
    
    # 1. Base Model: MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )

    # 2. Freeze the base to keep ImageNet features initially
    base_model.trainable = False

    # 3. Add Clinical Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Compress spatial features
    
    x = Dense(256, activation='relu')(x) # High-density feature layer
    x = BatchNormalization()(x)          # Stabilize training
    x = Dropout(0.5)(x)                  # Prevent overfitting to specific X-ray patterns
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(1, activation='sigmoid')(x) # Binary: 0=Normal, 1=Pneumonia

    # 4. Integrate into Model object
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    print("[SUCCESS] 🏥 Clinical Model Architecture initialized.")
    return model
