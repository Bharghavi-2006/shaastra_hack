# **shaastra\_hack**

## **Overview**

**ID Verifier Web App** is a web-based ID verification tool built with **Flask**, **Tesseract OCR**, and a **HuggingFace AI detector**.
You can upload images of IDs like **Aadhaar**, **PAN**, **Passport**, **Driving License**, or **Voter ID**, and the app will:

* Extract text using OCR
* Check ID formats for correctness
* Give an AI-based likelihood of the image being tampered or AI-generated

---

## **Features**

* **Instant Verification:** Upload any ID image and get results immediately.
* **Image Preprocessing:** Deskewing, denoising, and sharpening for better OCR accuracy.
* **OCR Text Extraction:** Extracts text from the ID and shows the first 500 characters.
* **Document Type Detection:** Identifies the ID type (Aadhaar, PAN, Passport MRZ, etc.) and checks its format.
* **AI-based Authenticity Check:** Flags potential fake or tampered IDs.
* **User-friendly Interface:** Clear display of verification status and reasons.

---

## **Setup Instructions**

1. **Clone the Project:** Copy the project folder to your local machine.

2. **Install Dependencies:**

```bash
pip install flask pillow numpy opencv-python pytesseract transformers torch
```

3. **Install Tesseract OCR:**

* **Windows:** [Download Installer](https://github.com/tesseract-ocr/tesseract)
* **Mac:** `brew install tesseract`
* **Linux:** `sudo apt install tesseract-ocr`

4. **Update Tesseract Path (if needed):**

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

5. **Run the App:**

```bash
python ocr_flask_app.py
```

6. **Open in Browser:** Visit `http://127.0.0.1:5000` and upload an ID image.

---

## **Folder Structure**

```
shaastra_hack/
├─ ocr_flask_app.py      # Main Flask + OCR + AI code
├─ uploads/              # Stores last preprocessed ID image
├─ templates/            # Optional HTML templates
├─ static/               # Optional CSS/JS files
└─ README.md             # This file
```

---

## **Notes**

* Works best with **clear, front-facing ID images**.
* OCR may struggle with **blurry, dark, or handwritten IDs**.
* AI detection provides a **likelihood score**, not a guarantee.
* Can be extended to support **new ID types or stricter checks**.

---

## **Future Improvements**

* Multi-language OCR support.
* Database validation of ID numbers.
* Mobile-friendly, drag-and-drop interface.
* Batch processing multiple IDs.
* Improved AI detection for photoshopped images.

---
