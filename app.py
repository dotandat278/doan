from flask import Flask, render_template,request,send_file,Response
import pickle
import soundfile
import numpy as np
import librosa 
import os
app = Flask(__name__)


UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Cái đoạn code sau đây các phương pháp rút trích đặc trưng
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
# Load hàm tiền xử lý dữ liệu. Cái hàm này được lấy từ trên quá trình huấn luyện ở file code google colab
model_path = "pickle/classifier.model"  
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
# Đây là đoạn code xử lý code xây dựng ứng dụng: nếu địa chỉ IP vào là 127.0.0:5000 là sẽ trả file html "Home"
@app.route("/")
def home():
    return render_template("Home.html")

# Đoạn code đường dẫn dự đoán. Nếu upload file lên và nhấn nút dự đoán thì sẽ chạy vào hàm này xử lý và check kết quả.
@app.route("/dudoan", methods=['POST'])
def dudoan():
    if request.method == 'POST':
        path_demo = request.files['file']
        try:
            feature_demo = extract_feature(path_demo, mfcc=True, chroma=True, mel=False, contrast=False, tonnetz=False)
            # Dùng mô hình đã huấn luyện để dự đoán
            y_pred_demo = model.predict([feature_demo])[0]
            print("Nhãn dự đoán: ", y_pred_demo)
            if y_pred_demo == "male":
                output = "Male"
            else:
                output = "Female"
            return render_template('Result.html',  data = [{"output":output}])
        except:
            return render_template("Error.html")

# Hàm main chạy chương trình web Flask trên local
if __name__ == "__main__":
    app.run()
