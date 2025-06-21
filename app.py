from flask import Flask, request, render_template, send_file
import sys
import os
from ultralytics import YOLO
import uuid
import webbrowser
from threading import Timer

app = Flask(__name__)

def resource_path(relative_path):
    """获取打包后或未打包时资源的绝对路径"""
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

model_path = resource_path("best.pt")
model = YOLO(model_path)
# 加载模型
# model = YOLO("best.pt")

# 上传和结果文件夹
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # 推理
            results = model.predict(
                filepath,
                iou=0.2,
                conf=0.5,
                show_labels=False,
                show_conf=False,
                line_width=3,
                nms=True,
                save=True,
                project=RESULT_FOLDER,
                name='pred',
                exist_ok=True
            )
            # 结果文件路径
            result_path = os.path.join(results[0].save_dir, filename)

            # 用 resource_path 转换绝对路径
            result_path = resource_path(result_path)

            print("Sending file:", result_path)
            print("Exists:", os.path.exists(result_path))

            return send_file(result_path, mimetype='image/jpeg')

    return render_template('index.html')


if __name__ == '__main__':
    port = 5000
    url = f"http://127.0.0.1:{port}/"

    def open_browser():
        webbrowser.open_new(url)

    # 只在主进程打开浏览器
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        Timer(1, open_browser).start()

    app.run(debug=True, port=port)
