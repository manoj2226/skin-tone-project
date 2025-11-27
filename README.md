# Skin Tone Classification Project (Deep Learning + Streamlit)

This project is a complete end-to-end **Skin Tone Classification System** built using  
**TensorFlow (MobileNetV2)** and **Streamlit UI**.

Users can upload an image, and the app predicts their **skin tone category**:
- Light  
- Mid-light  
- Mid-dark  
- Dark  

### 🔥 Features
- Deep Learning model using **MobileNetV2**
- Image preprocessing & classification
- Streamlit web interface
- Clean folder structure
- Color tone recommendations via `colorTones.json`

---

## 📁 Project Structure

skin-tone-project/
│
├── app_streamlit.py # Streamlit UI
├── predict.py # Prediction + recommendations
├── train.py # Model training script
├── models/ # Saved models (.keras / .h5)
├── dataset/ # Image dataset (ignored in Git)
├── colorTones.json # Makeup recommendations
├── requirements.txt
└── README.md


