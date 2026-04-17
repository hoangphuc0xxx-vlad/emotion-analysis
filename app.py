import os
import json
from flask import Flask, render_template, request
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import random
import os
import sys

# Ép hệ thống nhận diện ffmpeg.exe chứa bên trong thư mục AI_project
os.environ["PATH"] += os.pathsep + os.path.dirname(os.path.abspath(__file__))

from flask import Flask, render_template, request, redirect
from faster_whisper import WhisperModel

# LOAD API KEY
load_dotenv()
API_KEY = os.getenv("APIkey")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash") 

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 

# LOAD WHISPER MODEL (global )
# Buff model lên cứng cấp "small" để diệt gọn tiếng Việt, vượt xa "base"
whisper_model = WhisperModel("small", device="cpu")

# BIẾN THỐNG KÊ
thong_ke_cam_xuc = {
    'Tích cực': 0,
    'Tiêu cực': 0,
    'Trung Tính': 0,
    'tong_so': 0
}




# HÀM GỌI AI

def doc_text_tu_image(image_bytes):
    try:
        prompt = """
        Hãy đọc toàn bộ văn bản trong ảnh và trả về nguyên văn.
        Không giải thích, chỉ trả về text.
        """
        img = Image.open(io.BytesIO(image_bytes))
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print("Lỗi OCR:", e)
        return ""

def phantich_camxuc(text=None, image_bytes=None):

    prompt = f"""
Phân tích cảm xúc từ nội dung sau.

- Nếu có văn bản → dùng văn bản
- Nếu có hình ảnh → phân tích cảm xúc từ hình ảnh
- Nếu có cả 2 → kết hợp cả hai

Trả về JSON:
{{
  "label_name": "Tích cực/Tiêu cực/Trung Tính",
  "explanation": "giải thích ngắn",
  "score": {{
    "positive": số,
    "negative": số,
    "neutral": số
  }}
}}
"""

    parts = []

    if text:
        parts.append(prompt + f'\nVăn bản: "{text}"')
    else:
        parts.append(prompt)

    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            parts.append(img)
        except Exception as e:
            print("Lỗi đọc ảnh bằng PIL:", e)

    response = model.generate_content(
        parts,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
    )

    ai_text = response.text.strip()
    ai_text = ai_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(ai_text)
    except:
        print("AI trả lỗi:", ai_text)
        return {
            "label_name": "Trung Tính",
            "explanation": "Lỗi parse JSON",
            "score": {"positive": 33, "negative": 33, "neutral": 34}
    }


# TÍNH % THỐNG KÊ
def tinh_ty_le():
    tong = thong_ke_cam_xuc['tong_so']

    if tong == 0:
        return {
            'Tích cực': '0%',
            'Tiêu cực': '0%',
            'Trung Tính': '0%',
            'tong_so': 0,
            'so_luong': thong_ke_cam_xuc
        }

    return {
        'Tích cực': f"{round(thong_ke_cam_xuc['Tích cực'] / tong * 100, 1)}%",
        'Tiêu cực': f"{round(thong_ke_cam_xuc['Tiêu cực'] / tong * 100, 1)}%",
        'Trung Tính': f"{round(thong_ke_cam_xuc['Trung Tính'] / tong * 100, 1)}%",
        'tong_so': tong,
        'so_luong': thong_ke_cam_xuc
    }


# ROUTE
@app.route('/', methods=['GET', 'POST'])
def index():

    global thong_ke_cam_xuc

    ket_qua = {
        'status': 'Chờ nhập dữ liệu',
        'label': '---',
        'score': '0%',
        'text_cu': '',
        'explanation': ''
    }

    if request.method == 'POST':

        user_input = request.form.get('text-content', '').strip()

        file = request.files.get('image-file')

        image_bytes = None
        ocr_text = ""

        if file and file.filename != '':
            image_bytes = file.read()
            ocr_text = doc_text_tu_image(image_bytes)

        try:
            n = int(request.form.get('n-times', 100))
        except:
            n = 100

        n = max(1, min(10000, n))

        if user_input == "" and image_bytes is None:
            ket_qua['status'] = "nhập text hoặc image :)"

        else:
            try:
                if user_input:
                    final_text = user_input if user_input else ""
                elif ocr_text:
                    final_text = ocr_text
                else:
                    final_text = ""

                ai_data = phantich_camxuc(final_text, image_bytes)

                label = ai_data.get("label_name", "Trung Tính")
                explanation = ai_data.get("explanation", "")

                scores = ai_data.get("score", {})

                pos = float(scores.get("positive", 0))
                neg = float(scores.get("negative", 0))
                neu = float(scores.get("neutral", 0))

                # nếu AI trả dạng 0-1 → convert
                if pos <= 1 and neg <= 1 and neu <= 1:
                    pos *= 100
                    neg *= 100
                    neu *= 100

                total = pos + neg + neu

                if total == 0:
                    pos, neg, neu = 33.3, 33.3, 33.4
                else:
                    pos = round(pos / total * 100, 1)
                    neg = round(neg / total * 100, 1)
                    neu = round(100.0 - pos - neg, 1)

                weights = [pos, neg, neu]
                labels = ['Tích cực', 'Tiêu cực', 'Trung Tính']
                if sum(weights) == 0:
                    weights = [33.3, 33.3, 33.4]
                    
                # Áp dụng Thống kê vĩ mô (Monte Carlo) để quyết định kết quả cuối cùng thay vì tin vào lần đầu
                current_rolls = {'Tích cực': 0, 'Tiêu cực': 0, 'Trung Tính': 0}
                for i in range(n):
                    result = random.choices(labels, weights=weights, k=1)[0]
                    current_rolls[result] += 1
                    
                    # Vẫn tiếp tục cộng dồn vào máy chủ toàn phiên
                    thong_ke_cam_xuc[result] += 1
                    thong_ke_cam_xuc['tong_so'] += 1

                # Nhãn và điểm tin cậy dứt khoát thuộc về số đông (n) mô phỏng
                label = max(current_rolls, key=current_rolls.get)
                score_str = f"{round(current_rolls[label] / n * 100, 1)}%"
                
                pos = round(current_rolls['Tích cực'] / n * 100, 1)
                neg = round(current_rolls['Tiêu cực'] / n * 100, 1)
                neu = round(100.0 - pos - neg, 1)

                ket_qua = {
                    'status': f'Hoàn thành ({n} lần đối chiếu)',
                    'label': label,
                    'score': score_str,

                    'pos': pos,
                    'neg': neg,
                    'neu': neu,
                    'n': n,

                    # 'confidence': str(confidence) + "%",
                    'text_cu': final_text,
                    'explanation': explanation,
                    'ocr_text': ocr_text
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Lỗi:", e)
                ket_qua['status'] = f"Lỗi AI: {str(e)[:50]}..."

    return render_template(
        'index.html',
        data=ket_qua,
        thong_ke=tinh_ty_le()
    )

@app.route('/reset')
def reset():
    global thong_ke_cam_xuc
    thong_ke_cam_xuc={
        'Tích cực': 0,
        'Tiêu cực': 0,
        'Trung Tính': 0,
        'tong_so': 0
    }
    return redirect('/')

@app.route('/voice', methods=['POST'])
def voice():
    try:
        audio_file = request.files.get('audio')

        if not audio_file:
            return {"error": "No audio"}, 400

        audio_path = "audio_web.webm"
        audio_file.save(audio_path)

        import os
        print("==== VOICE ====")
        print("Size:", os.path.getsize(audio_path))

        try:
            # Ép buộc dịch chuẩn Tiếng Việt (language='vi') và chặn ảo giác lặp từ
            segments, _ = whisper_model.transcribe(audio_path, beam_size=5, language="vi", condition_on_previous_text=False)
            text = ""
            for seg in segments:
                text += seg.text

            print("TEXT:", text)

        except Exception as e:
            print("WHISPER ERROR:", e)
            import traceback
            traceback.print_exc()
            return {"error": "Lõi Whisper chết: " + str(e)}, 500

        ai_data = phantich_camxuc(text)

        return {
            "text": text,
            "label": ai_data.get("label_name"),
            "explanation": ai_data.get("explanation")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ROUTE VOICE ERROR:", e)
        return {"error": "Máy chủ xử lý thất bại: " + str(e)}, 500


if __name__ == '__main__':
    app.run(debug=True)