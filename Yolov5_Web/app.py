# coding=utf-8
import os
import cv2
import pymysql
import torch
from flask import Flask, request, render_template, redirect, flash, session, send_from_directory
from werkzeug.utils import secure_filename
from LipReading.ResNet import ResNet
from LipReading.DenseNet import Dense3D
from LipReading.data_process import _sample
from LipReading.opts import args
from LipReading.utils import padding_batch
from Yolov5_Web.one_test import yolov5

# Define a flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'lip_reading'

# Init
yolov5_det = yolov5()
model = ResNet(3, args.num_class)
# model = Dense3D(3, args.num_class)
# 读取模型和词表
model_path = 'lip_models/ResNet_model.pt'
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f, map_location=args.device))
model.eval()

def video_to_frames(path):
    """
    输入：path(视频文件的路径)
    """
    # VideoCapture视频读取类

    # 抽取帧数
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # filename = os.path.basename(path)  # 获取文件名
    # stem, suffix = os.path.splitext(filename)
    # save_path = join(args.save_img, stem)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    img_list = []
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        if i % 4 == 0:
            img_list.append(frame)

    print("视频切帧完成！")
    return img_list


# 根据预测得到的坐标进行裁剪
def cut_img(img_list):
    # 进行目标检测得到坐标点
    result = yolov5_det.detect(img_list)

    cut_img_list = []
    for idx, image in enumerate(img_list):
        labels = result[idx]
        cropped = image[int(labels[1]): int(labels[3]), int(labels[0]):int(labels[2])]
        cut_img_list.append(cropped)

    print("嘴型检测并裁剪完成！")
    return cut_img_list


def process(cut_img_list):
    data = []
    sample = _sample(cut_img_list, bilater=False)  # 预处理
    if sample is not None:
        data.append(sample)

    return data


def model_predict(model, cut_img_list, vocab_path, device):
    model.to(device)

    id2label = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    # 预处理
    test_data = process(cut_img_list)
    test_data = torch.tensor(padding_batch(test_data))
    print("数据预处理完成！")
    ##############################
    #            预测
    ##############################
    pre_result = []
    with torch.no_grad():
        batch_inputs = test_data.to(device)
        logist = model(batch_inputs)

        pred = torch.argmax(logist, dim=-1).tolist()
        pre_result.append(id2label[pred[0]])
    return pre_result


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 读取video文件
            f = request.files['file']

            # 保存前端读取的视频到uploads
            basepath = args.save_video
            file_path = os.path.join(basepath, secure_filename(f.filename))
            f.save(file_path)
            img_list = video_to_frames(file_path)
            cut_img_list = cut_img(img_list)
            del img_list

            vocab_path = 'lip_models/vocab100.txt'
            # 载入网络进行预测
            result = model_predict(model, cut_img_list, vocab_path, args.device)
            result = str(result[0])
            print("识别结果：" + result)
            return result
        except Exception as e:
            return "错误，无法正确识别"
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')#设置icon
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static/images'),#对于当前文件所在路径,比如这里是static下的favicon.ico
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 获取请求中的数据
        username = request.form.get('username')
        password = request.form.get('password')

        # 连接数据库，判断用户名+密码组合是否匹配
        conn = pymysql.connect(host="localhost", port=3306, user="root", passwd='root', db='lip_read')
        cur = conn.cursor()
        try:
            sql = " select username,password from sign where username='%s' and password='%s' " % (username, password)
            cur.execute(sql)
            data = cur.fetchall()  # 查询表数据
            data = sorted(data, reverse=True)

        except:
            flash('用户名或密码错误！')
            return render_template('login.html')
        finally:
            conn.close()

        if data != []:
            # 登录成功后存储session信息
            session['is_login'] = True
            session['name'] = username
            return redirect('/')
        else:
            flash('用户名或密码错误！')
            return render_template('login.html')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm')
        # 判断所有输入都不为空
        if username and password and confirm_password:
            if password != confirm_password:
                flash('两次输入的密码不一致！')
                return render_template('register.html', username=username)
            # 连接数据库
            conn = pymysql.connect(host="localhost", port=3306, user="root", passwd='root', db='lip_read')
            cur = conn.cursor()
            # 查询输入的用户名是否已经存在
            sql_same_user = " select username from sign where username='%s'" % (username)
            cur.execute(sql_same_user)
            data = cur.fetchall()  # 查询表数据
            data = sorted(data, reverse=True)
            if data != []:
                flash('用户名已存在！')
                return render_template('register.html', username=username)
            # 通过检查的数据，插入数据库表中
            cur.execute("insert into sign(username, password) values(%s, %s)", [username, password])
            conn.commit()
            conn.close()
            # 重定向到登录页面
            return redirect('/login')
        else:
            flash('所有字段都必须输入！')
            if username:
                return render_template('register.html', username=username)
            return render_template('register.html')
    return render_template('register.html')


@app.route('/logout')
def logout():
    # 退出登录，清空session
    if session.get('is_login'):
        session.clear()
    return redirect('/')


if __name__ == '__main__':
    app.run(port=5000)

