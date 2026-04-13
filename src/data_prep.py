import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, batch_size=32, target_size=(224, 224)):
    """
    Creates robust data pipelines for clinical image analysis.
    Incudes standard medical imaging augmentations.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # 1. Industrial-grade Data Augmentation (Training Only)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 2. Rescaling for Validation and Test (No augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"[INFO] 📂 Loading dataset from {data_dir}...")
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = None
    if os.path.exists(test_dir):
        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )

    return train_generator, val_generator, test_generator
