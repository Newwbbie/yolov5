 #!/usr/bin/python

from flask import Flask,jsonify,request,abort
from detect_service import run
import os
import json
import time
from pathlib import Path
import torch
import urllib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return '<h1>Home</h1>'

@app.route('/detect', methods=['POST'])
def detect():
    try:
        time1 = time.time()
        result = []
        params = json.loads(str(request.data, 'utf-8'))
        app.logger.info('参数：%s', params)

        user_id = str(params['userId'])
        uid_path = f'data/serviceImg/{user_id}/'
        imgs = params['imgUrlList']
        if not os.path.exists(uid_path):
            os.mkdir(uid_path)
        # imgs = request.files.getlist('imgs')
        for url in imgs:
            if url.startswith(('http:/', 'https:/')):
                # 保存到 serviceImg 文件夹
                file = Path(urllib.parse.unquote(url).split('?')[0]).name
                file_path = uid_path + file
                if Path(file_path).is_file():
                    print(f'找到 {url} 位于 {file_path}')  # file already exists
                else:
                    print(f'下载 {url} 到 {file_path}...')
                    torch.hub.download_url_to_file(url, file_path)
                    assert Path(file_path).exists() and Path(file_path).stat().st_size > 0, f'下载失败: {url}'  # check
        res = run(
            weights='runs/train/5x-fiberall-120-720/weights/best.pt',
            source=f'data/serviceImg/{user_id}/',  # file/dir/URL/glob, 0 for webcam
            data='data/fiber.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.1,  # confidence threshold
            iou_thres=0.2,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=True,  # save cropped prediction boxes
            nosave=True,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=True,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
        )
        # # 检测完之后删除缓存图片
        # os.remove(img_path)
        
        time2 = time.time()
        result = {
            'code': 200, 
            'time': f'{time2 - time1:.3f}s', 
            'msg': res
        }
        return json.dumps(result, ensure_ascii=False)
    except BaseException:
        app.logger.error("发生了异常")
        result = {
            'code': 400, 
            'msg': '发生了异常'
        }
        return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=30003,debug=True)