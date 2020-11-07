# -*- coding: utf-8 -*-
"""Gender_Classification_audio.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PLvNgYkohBPlbOLmQ1AcszxjR37wMAwo
"""

# Import các thư viện cần thiết cho chương trình
import soundfile # to read audio file
import librosa # to extract speech features
import glob, os
import pickle 
from sklearn.metrics import *
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

# Rút trích đặc trưng từ file audio.  Ở đây ta có 5 đặc trưng là : mfcc, chroma, melspectrogram, spectral_contrast
# tonnetz
# https://viblo.asia/p/so-luoc-ve-mel-frequency-cepstral-coefficients-mfccs-1VgZv1m2KAw
# https://viblo.asia/p/dac-trung-co-ban-ve-am-thanh-ORNZqjzMl0n
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

int2label = {"m": "male", "f": "female"}

def load_data(path_file):
    X, y = [], []
    dict_label = {}
    with open(path_file + "genders.txt","r",encoding="utf8") as file:
        lines = file.read().split("\n")
        for line in lines:
            line = line.strip()
            if line != "":
                dict_label[line.split(" ")[0]] = line.split(" ")[1]
    for file in glob.glob(path_file + "waves/*/*.wav"):
        print(file)
        # get the emotion label
        label = dict_label[file.split("\\")[1]]
        label = int2label[label]
        # rút trích các đặc trưng. Ở đây có tổng cộng 5 đặc trưng. dựa theo ví dụ dươi thì anh rút đặc trưng mfcc.
        # Nếu em muốn rút đăc trưng nào thì set biến nó bằng True. ví dụ đặc trưng chroma = True.
        # Em cứ chạy thử nghiệm riêng từng đặc trưng, rồi các đặc trưng kết hợp như mfcc vs chroma. chỉ cần set 2 đặc trưng anfy = True
        features = extract_feature(file, mfcc=True, chroma=True, mel=False, contrast=False, tonnetz=False)
        # Đưa đặc trưng rút trích danh sách X, nhãn tương uwngsg vào biến danh sách y
        X.append(features)
        y.append(label)

    return np.array(X), y

# Đọc dữ liệu tập train tập test và rút trích đặc trưng
path_train = "train/"
path_test =  "test/"
X_train, y_train = load_data(path_train)
X_test, y_test = load_data(path_test)

# Hiển thị một số thông tin về bộ dữ liệu
print("Số lượng mẫu trong tập huấn luyện:", X_train.shape[0])
print("Số lượng mẫu trong tập kiểm tra:", X_test.shape[0])
print("Số lượng đặc trưng rút trích:", X_train.shape[1])

# Huấn luyện mô hình
# Mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn_model.predict(X_test).tolist()
print(len(y_pred),len(y_test))
print("y_pred: ", y_pred)
print("y_test: ", y_test)
# tính toán accuracy và F1 score
print("Ket qua mo hinh KNN")

accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy*100))
print("F1-Score: {:.2f}%".format(f1score*100))

# Hiển thị kết quả chi tiết trên từng nhãn
print(classification_report(y_test,y_pred))
# Lưu mô hình lại cho quá trình xây dựng chương trình demo
pickle.dump(knn_model, open("KNN_classifier.model", "wb"))

# Khởi tạo mô hình SVM và huấn luyện trên tập dữ liệu train
svm_model = LinearSVC(C=1.0, intercept_scaling=1, max_iter=1000, random_state=42, verbose=0)
svm_model.fit(X_train, y_train)

# Dự đoán của mô hình SVM 
y_pred = svm_model.predict(X_test)
# Tính toán độ chính xác, f1-score của mô hình svm
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, average='weighted')

print("Ket qua SVM")

print("Accuracy: {:.2f}%".format(accuracy*100))
print("F1-Score: {:.2f}%".format(f1score*100))

# Dự đoán chi tiết trên từng nhãn
print(classification_report(y_test,y_pred))
# Lưu mô hình lại.
pickle.dump(svm_model, open("SVM_classifier.model", "wb"))

# Mô hình mạng nhân tạo - neural networks
model_params = { 'alpha': 0.01, 'batch_size': 256, 'epsilon': 1e-08, 'hidden_layer_sizes': (300,), 'learning_rate': 'adaptive', 'max_iter': 500, }
neural_model = MLPClassifier(**model_params)
neural_model.fit(X_train, y_train)

# Dự đoán của mô hình neural network
y_pred = neural_model.predict(X_test)

print("Ket qua mang neuro nhan tao")
# Tính kết quả trên độ đo. accuracy , precision, recall và f1-score. Search gg là ra.
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy*100))
print("F1-Score: {:.2f}%".format(f1score*100))

# Dự đoán chi tiết trên từng nhãn mô hình neural network
print(classification_report(y_test,y_pred))
# Lưu mô hình lại.
pickle.dump(neural_model, open("Neural_classifier.model", "wb"))

# Nghe âm thanh của ví dụ
path_demo = "test/waves/VIVOSDEV12/VIVOSDEV12_006.wav"

# Rút trích đặc trưng file demo.
feature_demo = extract_feature(path_demo, mfcc=True, chroma=True, mel=False, contrast=False, tonnetz=False)
# Dùng mô hình neural_network dự đoán.
y_pred_demo = neural_model.predict([feature_demo])
print("Nhãn dự đoán: ", y_pred_demo)

# Xóa dữ liệu. Em chỉ cần xóa nguyên folder hoặc từng file riêng lẻ folder là được. chứ đừng đổi tên folder nhé.
# Theo anh nghĩ em nên lấy 500 file male, 500 file female để train, và 200 male + 200 female để làm tập test.
# Bản nào nộp thầy thì xóa cmt thôi là dc.