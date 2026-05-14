# 🐾 DNNetV3: Dog Nose Biometrics User Guide

Welcome to the **DNNetV3** Dog Nose Biometric System. This application uses State-of-the-Art (SOTA) Artificial Intelligence to identify dogs based on the unique ridge patterns on their noses—essentially a "fingerprint" for dogs.

---

## 🛠️ 1. Prerequisites
Before running the app, ensure you have Python installed (version 3.8 or higher). You will also need to install the required AI libraries.

Open your terminal (Command Prompt or PowerShell) and run:
```bash
pip install torch torchvision timm opencv-python numpy Pillow streamlit
```

## 📂 2. File Setup
Ensure the following files are in the **same folder**:
1. `det5.py`: The AI engine (contains the model and logic).
2. `app.py`: The user interface (the dashboard you see in your browser).
3. `dog_biometrics.db`: (Optional) This will be created automatically to store your data.

## 🚀 3. How to Run
1. Open your terminal in the project folder.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. A new tab will open in your web browser (usually at `http://localhost:8501`).

---

## 📖 4. Using the App

### 🔍 Identify Dog
*   **What it does:** Compares a new photo against all dogs you have already registered.
*   **How to use:** Upload a clear, close-up photo of a dog's nose. The app will show you the most likely match and a "Confidence Score."
*   **Pro Tip:** If the "Sharpness Score" is low, try taking the photo again in better lighting.

### 📝 Register New Dog
*   **What it does:** Saves a dog's nose "fingerprint" into the database.
*   **How to use:** Enter the dog's name and upload a high-quality nose photo. The AI will extract a unique 1024-digit code (embedding) that represents that dog.

### ⚙️ Manage Database
*   **What it does:** Allows you to view who is registered and delete entries.
*   **Danger Zone:** You can wipe the entire database by typing `WIPE` in the confirmation box.

---

## ⚠️ 5. Troubleshooting (Common Errors)

### "OperationalError: no such column: registered_at"
If you see this error, it means you have an old version of the database file. 
*   **Fix:** Simply delete the file named `dog_biometrics.db` in your folder and restart the app. It will recreate a fresh, updated database.

### "ModuleNotFoundError: No module named 'timm'"
*   **Fix:** Run `pip install timm` in your terminal.

### Accuracy is low?
*   Ensure the nose is centered in the photo.
*   Avoid photos with heavy saliva or dirt on the nose.
*   The AI works best on "Macro" (close-up) shots.
