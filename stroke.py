import pickle
import streamlit as st
import numpy as np
import pandas as pd
import time

# Load model
stroke_model = pickle.load(open('stroke_model_random.sav', 'rb'))

# Initialize user state
if 'user_state' not in st.session_state:
    st.session_state.user_state = {
        'username': '',
        'password': '',
        'logged_in': False
    }

if 'page' not in st.session_state:
    st.session_state.page = None  # Set default page to None

# Fungsi untuk logout
def logout():
    st.session_state.user_state['logged_in'] = False
    st.session_state.user_state['username'] = ''
    st.session_state.page = None  # Reset to None
    # st.rerun()

# Tampilan login
if not st.session_state.user_state['logged_in']:
    st.header('Welcome To App NeuroScan')
    st.write('Please login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    submit = st.button('Login')

    # Check if user is logged in
    if submit:
        if username == 'admin' and password == '1234':
            st.session_state.user_state['username'] = username
            st.session_state.user_state['password'] = password
            st.session_state.user_state['logged_in'] = True
            st.session_state.page = 'welcome'  # Set to welcome page after login
            st.rerun()  # Rerun to show the main app
        else:
            st.write('Invalid username or password')

# Tampilan aplikasi jika sudah login
if st.session_state.user_state['logged_in']:
    st.sidebar.markdown("<h1 style='text-align: justify; font-size: 2em;'>🧠NeuroScan</h1>", unsafe_allow_html=True)

     # Menambahkan teks di bawah judul
    st.sidebar.markdown("<h5 style='text-align: justify;'>NeuroScan adalah aplikasi berbasis machine learning untuk mendeteksi risiko stroke. Aplikasi ini menggunakan model Random Forest untuk memprediksi kemungkinan stroke berdasarkan data pasien.</h5>", unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.button("Random Forest", on_click=lambda: setattr(st.session_state, 'page', 'random_forest'))
    st.sidebar.button("About", on_click=lambda: setattr(st.session_state, 'page', 'about'))
    st.sidebar.button("Logout", on_click=logout)


    # Menampilkan pesan sambutan
    welcome_placeholder = st.empty()  # Initialize the placeholder outside of the condition
    if st.session_state.page == 'welcome':
        welcome_placeholder.markdown(
            "<h1 style='text-align: center; font-size: 4em; color: #fefcfc;'>Welcome to NeuroScan 🧠</h1>",
            unsafe_allow_html=True
        )
    else:
        welcome_placeholder.empty()  # Clear the welcome message if not on the welcome page

    # Menampilkan konten About
    def about():
        st.title("NeuroScan 🧠")
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

    # Halaman Random Forest
    def random_forest():
        st.title('Aplikasi Deteksi Stroke')

        # Membagi Kolom
        col1, col2 = st.columns(2)

        with col1:
            # Gender
            st.header(":blue[Gender]")
            gender = st.radio('Pilih Jenis Kelamin', (1, 0))
            keterangan = 'Perempuan' if gender == 1 else 'Laki-Laki'
            st.write('Anda memilih:', keterangan)

            # Hypertension
            st.header(":blue[Hypertension]")
            hypertension = st.radio('Pilih Nilai Hypertension', (1, 0))
            keterangan = 'Yes' if hypertension == 1 else 'No'
            st.write('Anda memilih:', keterangan)

            # Ever Married
            st.header(":blue[Ever Married]")
            ever_married = st.radio('Pilih Nilai Ever Married', (1, 0))
            keterangan = 'Yes' if ever_married == 1 else 'No'
            st.write('Anda memilih:', keterangan)

            # Residence Type
            st.header(':blue[Residence Type]')
            Residence_type = st.radio('Input Nilai Residence Type', (1, 0))
            keterangan = 'Urban' if Residence_type == 1 else 'Rural'
            st.write('Anda memilih:', keterangan)

            # BMI
            st.header(':blue[BMI]')
            bmi = st.number_input('Input Nilai BMI')
            st.write("BMI yang di Masukkan ", bmi)

        with col2:
            # AGE
            st.header(':blue[AGE]')
            age = st.number_input('Masukkan Umur')
            st.write("Umur yang di Masukkan ", age)

            # Heart Disease
            st.header(":blue[Heart Disease]")
            heart_disease = st.radio('Pilih Nilai Heart Disease', (1, 0))
            keterangan = 'Yes' if heart_disease == 1 else 'No'
            st.write('Anda memilih:', keterangan)

            # Work Type
            st.header(':blue[Work Type]')
            work_type = st.radio('Input Nilai Work Type', (1, 2, 3, 0))
            if work_type == 0:
                keterangan = 'Private'
            elif work_type == 1:
                keterangan = 'Govt Job'
            elif work_type == 2:
                keterangan = 'Self-employed'
            elif work_type == 3:
                keterangan = 'Children'
            st.write('Anda memilih:', keterangan)

            # AVG Glucose Level
            st.header(':blue[AVG Glucose Level]')
            avg_glucose_level = st.number_input('Input Nilai AVG Glucose Level')
            st.write("AVG Glucose level yang di Masukkan ", avg_glucose_level)

            # Smoking Status
            st.header(':blue[Smoking Status]')
            smoking_status = st.radio('Input Nilai Smoking Status', (1, 2, 3, 0))
            if smoking_status == 0:
                keterangan = 'Never Smoked'
            elif smoking_status == 1:
                keterangan = 'Formerly Smoked'
            elif smoking_status == 2:
                keterangan = 'Smokes'
            elif smoking_status == 3:
                keterangan = 'Unknown'
            st.write('Anda memilih:', keterangan)

        # code untuk prediksi
        stroke_diagnosis = ''

        # membuat tombol untuk prediksi
        if st.button('Test Prediksi Stroke'):
            stroke_prediction = stroke_model.predict([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]])

            placeholder = st.empty()
            placeholder.progress(0, "Menganalisa...")
            time.sleep(1)
            placeholder.progress(100, "Menganalisa...")
            time.sleep(1)
            placeholder.empty()

            stroke_diagnosis = 'Pasien Terdeteksi Stroke' if stroke_prediction[0] == 1 else 'Pasien Tidak Terdeteksi Stroke'
            st.success(stroke_diagnosis)

    # Pengaturan navigasi halaman
    if st.session_state.page == 'random_forest':
        random_forest()
    elif st.session_state.page == 'about':
        about()
