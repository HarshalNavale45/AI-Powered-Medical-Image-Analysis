import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(model, train_generator, val_generator, epochs=20):
    """
    Trains the clinical model with advanced feedback loops.
    """
    os.makedirs("models", exist_ok=True)
    
    # 1. Save only the best model weights
    checkpoint = ModelCheckpoint(
        "models/best_medical_model.keras", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    # 2. Stop early if model stops learning (prevents overfitting)
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    # 3. Dynamic Learning Rate - reduce LR if validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )

    print("[INFO] 🚀 Starting Clinical Model Training Phase...")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    print("[SUCCESS] 📈 Training Complete. Model optimized for clinical inference.")
    return history
