import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

# Fungsi untuk melatih model Random Forest dengan penyeimbangan kelas dan normalisasi
def train_random_forest(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_depth=None, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    return model, accuracy, report, scaler, cm

# Inisialisasi state user
if 'user_state' not in st.session_state:
    st.session_state.user_state = {
        'username': '',
        'password': '',
        'logged_in': False
    }

if 'page' not in st.session_state:
    st.session_state.page = 'main'  # Nilai default halaman utama

# Fungsi untuk logout
def logout():
    st.session_state.user_state['logged_in'] = False
    st.session_state.user_state['username'] = ''
    st.session_state.user_state['password'] = ''

# Fungsi login
def login():
    st.header('Welcome to NeuroScan App')
    st.write('Please login to continue')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    submit = st.button('Login')

    if submit:
        if username == 'admin' and password == '1234':
            st.session_state.user_state['username'] = username
            st.session_state.user_state['password'] = password
            st.session_state.user_state['logged_in'] = True
            st.rerun()  # Memastikan halaman di-refresh setelah login
        else:
            st.error('Invalid username or password')

# Fungsi utama aplikasi Streamlit setelah login
def main():
    # Menampilkan nama aplikasi dengan markdown
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 2em;'>🧠NeuroScan</h1>", unsafe_allow_html=True)

    # Menampilkan pesan sambutan
    welcome_placeholder = st.empty()
    welcome_placeholder.markdown(
        "<h1 style='text-align: center; font-size: 4em; color: #fefcfc;'>Welcome to NeuroScan 🧠</h1>",
        unsafe_allow_html=True
    )

    # Sidebar untuk upload file
    st.sidebar.header("Upload Dataset")
    data_file = st.sidebar.file_uploader("Upload CSV data (dataset stroke)", type=["csv"])

    # Membuat dua kolom untuk tombol About dan Logout
    col_about, col_logout = st.sidebar.columns(2)
    with col_about:
        if st.button("About"):
            st.session_state.page = 'about'
            st.rerun()

    with col_logout:
        if st.button("Logout"):
            logout()
            st.rerun()  # Memastikan halaman di-refresh setelah logout

    if data_file is not None:
        welcome_placeholder.empty()
        data = pd.read_csv(data_file)
        st.title("Stroke Detection using Random Forest")
        st.write("Data yang diupload:")
        st.write(data.head())

        # Asumsi kolom-kolom yang umum pada dataset stroke
        default_features = ["gender", "age", "hypertension", "heart_disease", "ever_married", 
                            "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
        
        target = "stroke" if "stroke" in data.columns else st.selectbox("Pilih kolom target untuk stroke", data.columns)
        
        features = st.multiselect("Pilih fitur", [col for col in data.columns if col != target], default=default_features)

        # Melatih model jika belum dilatih atau jika tombol "Train Model" ditekan
        if "model" not in st.session_state or st.button("Train Model"):
            X = data[features]
            y = data[target]
            model, accuracy, report, scaler, cm = train_random_forest(X, y)

            # Menyimpan model, scaler, dan matriks kebingungan
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.accuracy = accuracy
            st.session_state.report = report
            st.session_state.confusion_matrix = cm
            st.success("Model berhasil dilatih!")

        # Menampilkan matriks kebingungan untuk memantau performa pada kelas minoritas
        st.write(f"Akurasi model: {st.session_state.accuracy:.2f}")
        st.text("Classification Report:")
        st.text(st.session_state.report)
        if "confusion_matrix" in st.session_state:
            st.write("Confusion Matrix (Matriks Kebingungan):")
            st.write(st.session_state.confusion_matrix)
            
            # Visualisasi heatmap matriks kebingungan
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            ax.set_title("Confusion Matrix Heatmap")
            st.pyplot(fig)

            # Prediksi manual
            st.header("Masukkan Nilai Fitur untuk Prediksi")
            
            # Mendapatkan input dari pengguna
            col1, col2 = st.columns(2)
            input_data_dict = {}
            with col1: 
                if 'gender' in features:
                    gender = st.radio('Jenis Kelamin', ('perempuan', 'laki-laki'))
                    
                    # Convert gender to numeric
                    gender_numeric = 0 if gender == 'perempuan' else 1
                    
                    st.write('Anda memilih:', 'Perempuan (Female)' if gender_numeric == 0 else 'Laki-Laki (Male)')
                else:
                    gender_numeric = 0  # Placeholder if the feature is not selected
                
                input_data_dict['gender'] = gender_numeric

                if 'hypertension' in features:
                    hypertension = st.radio('Hypertension (Hipertensi)', ('Yes (Ya)', 'No (Tidak)'))
                    hypertension_numeric = 1 if hypertension == 'Yes (Ya)' else 0
                    st.write('Anda memilih:', hypertension)
                else:
                    hypertension_numeric = 0
                input_data_dict['hypertension'] = hypertension_numeric

                if 'ever_married' in features:
                    ever_married = st.radio('Pernah Menikah (Ever Married)', ('Yes (Ya)', 'No (Tidak)'))
                    ever_married_numeric = 1 if ever_married == 'Yes (Ya)' else 0
                    st.write('Anda memilih:', ever_married)
                else:
                    ever_married_numeric = 0
                input_data_dict['ever_married'] = ever_married_numeric

                if 'Residence_type' in features:
                    Residence_type = st.radio('Tipe Tempat Tinggal (Residence Type)', ('Urban (Perkotaan)', 'Rural (Pedesaan)'))
                    Residence_type_numeric = 1 if Residence_type == 'Urban (Perkotaan)' else 0
                    st.write('Anda memilih:', Residence_type)
                else:
                    Residence_type_numeric = 0
                input_data_dict['Residence_type'] = Residence_type_numeric

                if 'bmi' in features:
                    bmi = st.number_input('BMI')
                else:
                    bmi = 0
                input_data_dict['bmi'] = bmi

            with col2:
                if 'age' in features:
                    age = st.number_input('Age (Umur)', min_value=0.0)
                else:
                    age = 0
                input_data_dict['age'] = age

                if 'heart_disease' in features:
                    heart_disease = st.radio('Heart Disease (Penyakit Jantung)', ('Yes (Ya)', 'No (Tidak)'))
                    heart_disease_numeric = 1 if heart_disease == 'Yes (Ya)' else 0
                    st.write('Anda memilih:', heart_disease)
                else:
                    heart_disease_numeric = 0
                input_data_dict['heart_disease'] = heart_disease_numeric

                if 'work_type' in features:
                    work_type = st.radio('Work Type (Jenis Pekerjaan)', ('Private (Pribadi)', 'Govt Job (Pekerjaan Pemerintah)', 'Self-employed (Wiraswasta)', 'Children (Anak-anak)'))
                    work_type_numeric = ['Private (Pribadi)', 'Govt Job (Pekerjaan Pemerintah)', 'Self-employed (Wiraswasta)', 'Children (Anak-anak)'].index(work_type)
                    st.write('Anda memilih:', work_type)
                else:
                    work_type_numeric = 0
                input_data_dict['work_type'] = work_type_numeric

                if 'avg_glucose_level' in features:
                    avg_glucose_level = st.number_input('Avg Glucose Level (Rata-rata Glukosa)')
                else:
                    avg_glucose_level = 0
                input_data_dict['avg_glucose_level'] = avg_glucose_level

                if 'smoking_status' in features:
                    smoking_status = st.radio('Smoking Status (Status Merokok)', ('Never Smoked (Tidak Pernah Merokok)', 'Formerly Smoked (Sebelumnya Merokok)', 'Smokes (Merokok)', 'Unknown (Tidak Diketahui)'))
                    smoking_status_numeric = ['Never Smoked (Tidak Pernah Merokok)', 'Formerly Smoked (Sebelumnya Merokok)', 'Smokes (Merokok)', 'Unknown (Tidak Diketahui)'].index(smoking_status)
                    st.write('Anda memilih:', smoking_status)
                else:
                    smoking_status_numeric = 0
                input_data_dict['smoking_status'] = smoking_status_numeric



            # Mengonversi input menjadi format yang dapat digunakan model
        if st.button('Test Prediksi Stroke'):
            input_data = np.array([[input_data_dict[feature] for feature in features]])
            input_data_scaled = st.session_state.scaler.transform(input_data)
            prediction = st.session_state.model.predict(input_data_scaled)

            # Menampilkan hasil prediksi
            placeholder = st.empty()
            placeholder.progress(0, "Menganalisa...")
            time.sleep(1)
            placeholder.progress(100, "Menganalisa...")
            time.sleep(1)
            placeholder.empty()

            stroke_diagnosis = 'Pasien Terdeteksi Stroke' if prediction[0] == 1 else 'Pasien Tidak Terdeteksi Stroke'
            st.success(stroke_diagnosis)

# Menampilkan konten About
def about():
    st.title("NeuroScan")
    st.markdown("""
    <div style="text-align: justify;">
    NeuroScan adalah aplikasi berbasis machine learning untuk mendeteksi risiko stroke. 
    Aplikasi ini menggunakan model **Random Forest** untuk memprediksi kemungkinan stroke berdasarkan data pasien. 
    Kata "Neuro" secara langsung merujuk pada sistem saraf dan otak, yang merupakan pusat dari segala aktivitas neurologis. 
    Dalam konteks stroke, yang merupakan kondisi yang mempengaruhi aliran darah ke otak, nama ini sangat tepat karena menunjukkan bahwa aplikasi ini berkaitan dengan kesehatan otak dan sistem saraf. 
    Kata "Scan" mencerminkan fungsi utama aplikasi yang bertujuan untuk melakukan deteksi dan analisis. 
    Dalam konteks stroke, aplikasi ini mungkin menggunakan metode pemindaian data, analisis gejala, atau pemantauan parameter vital untuk mendeteksi kemungkinan serangan stroke dengan cepat dan akurat.
    Stroke merupakan keadaan darurat medis yang membutuhkan respon cepat. Nama NeuroScan dapat memberikan kesan bahwa aplikasi ini dirancang untuk membantu pengguna mengenali tanda-tanda stroke secara dini, sehingga memungkinkan tindakan cepat yang bisa menyelamatkan nyawa.
    Dengan menggunakan istilah yang terdengar modern dan teknis, NeuroScan juga menciptakan citra aplikasi yang inovatif dan berbasis teknologi, menggambarkan komitmen untuk menyediakan solusi kesehatan yang efisien dan efektif.
    </div>
    
    <h3 style="text-align: justify;">Tujuan</h3>
    <div style="text-align: justify;">
    NeuroScan membantu dalam deteksi dini dan tindakan preventif terhadap stroke, 
    menyediakan alat yang dapat membantu pengguna mendeteksi gejala stroke secara cepat dan akurat, 
    sehingga meningkatkan peluang penanganan yang efektif dan menyelamatkan nyawa.
    </div>

    <h3 style="text-align: justify;">Cara Kerja</h3>
    <div style="text-align: justify;">
    <ul>
        <li><strong>Upload Data</strong>: Unggah dataset dalam format CSV.</li>
        <li><strong>Prediksi</strong>: Aplikasi memprediksi risiko stroke menggunakan model yang sudah dilatih.</li>
        <li><strong>Akurasi</strong>: Model dilatih dengan dataset berimbang untuk akurasi prediksi yang lebih baik.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Back to Main Page"):
        st.session_state.page = 'main'
        st.rerun()

# Pengaturan navigasi halaman
if not st.session_state.user_state['logged_in']:
    login()
else:
    if st.session_state.page == 'main':
        main()
    elif st.session_state.page == 'about':
        about()
