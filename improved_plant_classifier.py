import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import drive

# ===========================
# 1. Google Drive 마운트
# ===========================
drive.mount('/content/drive')

# ===========================
# 2. 데이터셋 압축 해제
# ===========================
!unzip -o "/content/drive/MyDrive/plant_dataset.zip" -d "/content/dataset"

# ===========================
# 3. 데이터 준비 및 설정
# ===========================
data_dir = "/content/dataset/plant_dataset"
batch_size = 32
img_height = 224
img_width = 224

# 훈련 데이터셋
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 검증 데이터셋
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

print(f"클래스 개수: {num_classes}")
print(f"클래스 이름: {class_names}")

# ===========================
# 4. 클래스별 이미지 개수 확인
# ===========================
print("\n클래스별 이미지 분포:")
for class_name in class_names:
    class_path = os.path.join(data_dir, class_name)
    if os.path.exists(class_path):
        num_images = len(os.listdir(class_path))
        print(f"  {class_name}: {num_images}개")

# ===========================
# 5. 데이터 파이프라인 최적화
# ===========================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ===========================
# 6. 강화된 데이터 증강
# ===========================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# ===========================
# 7. 베이스 모델 로드 (MobileNetV2)
# ===========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# 베이스 모델 가중치 동결
base_model.trainable = False

print(f"\n베이스 모델 레이어 수: {len(base_model.layers)}")

# ===========================
# 8. 전체 모델 구축
# ===========================
model = Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# ===========================
# 9. 모델 컴파일
# ===========================
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ===========================
# 10. 콜백 설정
# ===========================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        '/content/drive/MyDrive/best_model_initial.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# ===========================
# 11. 초기 학습 (베이스 모델 동결)
# ===========================
print("\n" + "="*50)
print("1단계: 초기 학습 시작 (베이스 모델 동결)")
print("="*50)

epochs_initial = 15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_initial,
    callbacks=callbacks
)

# ===========================
# 12. 학습 결과 시각화
# ===========================
def plot_training_history(history, title_suffix=""):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(14, 5))
    
    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='훈련 정확도')
    plt.plot(epochs_range, val_acc, label='검증 정확도')
    plt.legend(loc='lower right')
    plt.title(f'훈련 및 검증 정확도{title_suffix}')
    plt.xlabel('에포크')
    plt.ylabel('정확도')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='훈련 손실')
    plt.plot(epochs_range, val_loss, label='검증 손실')
    plt.legend(loc='upper right')
    plt.title(f'훈련 및 검증 손실{title_suffix}')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history, " (1단계)")

# ===========================
# 13. 미세 조정 (Fine-tuning)
# ===========================
print("\n" + "="*50)
print("2단계: 미세 조정 시작 (일부 레이어 동결 해제)")
print("="*50)

# 베이스 모델 동결 해제
base_model.trainable = True

# 처음 100개 레이어는 계속 동결
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"동결된 레이어: 0 ~ {fine_tune_at-1}")
print(f"학습 가능한 레이어: {fine_tune_at} ~ {len(base_model.layers)-1}")

# 낮은 학습률로 재컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# 미세 조정용 콜백
callbacks_fine = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        '/content/drive/MyDrive/best_model_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 미세 조정 학습
epochs_fine = 10

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs_fine,
    callbacks=callbacks_fine
)

# 미세 조정 결과 시각화
plot_training_history(history_fine, " (2단계 - 미세 조정)")

# ===========================
# 14. 전체 학습 과정 통합 시각화
# ===========================
# 두 학습 과정을 하나의 그래프로
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(14, 5))

# 정확도
plt.subplot(1, 2, 1)
plt.plot(acc, label='훈련 정확도')
plt.plot(val_acc, label='검증 정확도')
plt.axvline(x=len(history.history['accuracy']), color='r', linestyle='--', label='미세 조정 시작')
plt.legend(loc='lower right')
plt.title('전체 학습 과정 - 정확도')
plt.xlabel('에포크')
plt.ylabel('정확도')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3)

# 손실
plt.subplot(1, 2, 2)
plt.plot(loss, label='훈련 손실')
plt.plot(val_loss, label='검증 손실')
plt.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='미세 조정 시작')
plt.legend(loc='upper right')
plt.title('전체 학습 과정 - 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===========================
# 15. 모델 평가
# ===========================
print("\n" + "="*50)
print("최종 모델 평가")
print("="*50)

loss, accuracy = model.evaluate(val_ds)
print(f"\n검증 손실: {loss:.4f}")
print(f"검증 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ===========================
# 16. 예측 결과 시각화
# ===========================
def plot_predictions(model, dataset, class_names, num_images=9):
    plt.figure(figsize=(12, 12))
    for images, labels in dataset.take(1):
        predictions = model.predict(images)
        pred_classes = np.argmax(predictions, axis=1)
        
        for i in range(min(num_images, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            
            predicted_class = class_names[pred_classes[i]]
            true_class = class_names[labels[i]]
            confidence = predictions[i][pred_classes[i]] * 100
            
            color = "green" if predicted_class == true_class else "red"
            plt.title(f"예측: {predicted_class} ({confidence:.1f}%)\n실제: {true_class}", 
                     color=color, fontsize=10)
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()

print("\n예측 결과 시각화:")
plot_predictions(model, val_ds, class_names)

# ===========================
# 17. Confusion Matrix 생성
# ===========================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 전체 검증 데이터에 대한 예측
y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Confusion Matrix 시각화
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('실제 클래스')
plt.xlabel('예측 클래스')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 분류 리포트 출력
print("\n" + "="*50)
print("분류 리포트")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

# ===========================
# 18. 최종 모델 저장
# ===========================
model.save('/content/drive/MyDrive/plant_classifier_final.keras')
print("\n최종 모델이 '/content/drive/MyDrive/plant_classifier_final.keras'에 저장되었습니다.")

print("\n" + "="*50)
print("학습 완료!")
print("="*50)