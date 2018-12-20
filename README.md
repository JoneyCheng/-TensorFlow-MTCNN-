＃-TensorFlow-MTCNN-
＃48*48样本生成
import os
import numpy as np
from PIL import Image
from MTCNN_TensorFlow.tools import gen_until

landmarks_FILE = r"E:\pycharm_project\datasets\list_landmarks_celeba.txt"

IMAGE_DIR = r"E:\pycharm_project\datasets\img_celeba_ALL"
ANNO_FILE = r"E:\pycharm_project\datasets\list_bbox_celeba_ALL.txt"

DST_IMAGE_DIR = r"E:\pycharm_project\datasets\48_train"
DST_ANNO_FILE = r"E:\pycharm_project\datasets\48_train.txt"


count = 0
floats = [0.1,0.15,0.4,0.6,0.7,0.75,0.8,0.85,0.9,0.95]
with open(DST_ANNO_FILE,"w") as dst_f:#打开要记录生成label的文件，并开始写入记录

    with open(ANNO_FILE) as f:#打开目前的label文件
        for i, line in enumerate(f):#遍历txt文件的每一行
            if i > 300000:#当遍历至100行的时候停止
                break

            if i >= 2:#从第二行开始处理

                strs = line.split(" ")#用空格切割字符串，得到文件名和坐标信息
                strs = list(filter(lambda x: bool(x), strs))#过滤掉多余的空格，只留下文字信息
                img_filename = strs[0].strip()#获得切割后的第[0]个元素（也就是文件名） 用变量接收
                img_path = os.path.join(IMAGE_DIR, img_filename)#用路径目录和文件名生成文件的（路径目录+文件名）
                # print(img_path)

                x1 = int(strs[1].strip())#同理保存x,y,h,w，加strip，去掉换行，转化成int类型
                y1 = int(strs[2].strip())
                w = int(strs[3].strip())
                h = int(strs[4].strip())

                x2 = x1 + w
                y2 = y1 + h#换算成坐标

                box = np.array([x1, y1, x2, y2])#用一个列表保存这个标签的坐标信息
                # print(box)

                cx = x1 + w // 2#将X,Y换算成坐标的中心点
                cy = y1 + h // 2

                side_len = max(abs(w), abs(h))#取最大的边，（正方形）

                with open(landmarks_FILE) as f:  # 打开目前的label文件
                    j = i
                    for j, line in enumerate(f):  # 遍历txt文件的每一行
                        if j > 300000:  # 当遍历至100行的时候停止
                            break
                        if j == i:  # 从第二行开始处理
                            strs = line.split(" ")  # 用空格切割字符串，得到文件名和坐标信息
                            strs = list(filter(lambda x: bool(x), strs))  # 过滤掉多余的空格，只留下文字信息
                            lefteye_x = int(strs[1].strip())
                            lefteye_y = int(strs[2].strip())
                            righteye_x = int(strs[3].strip())
                            righteye_y = int(strs[4].strip())
                            nose_x = int(strs[5].strip())
                            nose_y = int(strs[6].strip())
                            leftmouth_x = int(strs[7].strip())
                            leftmouth_y = int(strs[8].strip())
                            rightmouth_x = int(strs[9].strip())
                            rightmouth_y = int(strs[10].strip())

                            seed = floats[np.random.randint(0,len(floats))]#随机种子

                            for _ in range(10):#每张图截取10张

                                _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))#，偏移边长，最大的边长再加上或减去一个随机系数
                                _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))#偏移中心点X
                                _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))#偏移中心点Y

                                _x1 = _cx - _side_len / 2  # 偏移后的中心点换算回偏移后起始点X,Y
                                _y1 = _cy - _side_len / 2
                                _x2 = _x1 + _side_len  # 获得偏移后的X2,Y2
                                _y2 = _y1 + _side_len
                                #偏移后的的坐标点对应的是正方形

                                x_lens = _cx - cx
                                y_lens = _cy - cy

                                # _lefteye_x = lefteye_x + x_lens
                                # _lefteye_y = lefteye_y + y_lens
                                # _righteye_x = righteye_x + x_lens
                                # _righteye_y = righteye_y + y_lens
                                # _nose_x = nose_x + x_lens
                                # _nose_y = nose_y + y_lens
                                # _leftmouth_x = leftmouth_x + x_lens
                                # _leftmouth_y = leftmouth_y + y_lens
                                # _rightmouth_x = rightmouth_x + x_lens
                                # _rightmouth_y = rightmouth_y + y_lens

                                offset_x1 = (x1 - _x1) / _side_len#获得换算后的偏移率
                                offset_y1 = (y1 - _y1) / _side_len
                                offset_x2 = (x2 - _x2) / _side_len
                                offset_y2 = (y2 - _y2) / _side_len

                                offset_lefteye_x = (lefteye_x - _x1) / _side_len
                                offset_lefteye_y = (lefteye_y - _y1) / _side_len
                                offset_righteye_x = (righteye_x - _x1) / _side_len
                                offset_righteye_y = (righteye_y - _y1) / _side_len
                                offset_nose_x = (nose_x - _x1) / _side_len
                                offset_nose_y = (nose_y - _y1) / _side_len
                                offset_leftmouth_x = (leftmouth_x - _x1) / _side_len
                                offset_leftmouth_y = (leftmouth_y - _y1) / _side_len
                                offset_rightmouth_x = (rightmouth_x - _x1) / _side_len
                                offset_rightmouth_y = (rightmouth_y - _y1) / _side_len

                                # offset_lefteye_x = (lefteye_x - _lefteye_x) / _side_len
                                # offset_lefteye_y = (lefteye_y - _lefteye_y) / _side_len
                                # offset_righteye_x = (righteye_x - _righteye_x) / _side_len
                                # offset_righteye_y = (righteye_y - _righteye_y) / _side_len
                                # offset_nose_x = (nose_x - _nose_x) / _side_len
                                # offset_nose_y = (nose_y - _nose_y) / _side_len
                                # offset_leftmouth_x = (leftmouth_x - _leftmouth_x) / _side_len
                                # offset_leftmouth_y = (leftmouth_y - _leftmouth_y) / _side_len
                                # offset_rightmouth_x = (rightmouth_x - _rightmouth_x) / _side_len
                                # offset_rightmouth_y = (rightmouth_y - _rightmouth_y) / _side_len

                                with Image.open(img_path) as img:#打开路径下的原图片文件
                                    img_w, img_h = img.size#获得宽高

                                    if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:#判断偏移超出整张图片的就跳过，不截图
                                        continue

                                    crop_box = [_x1, _y1, _x2, _y2]#获得需要截取图片样本的坐标
                                    crop_box =np.array ([crop_box])#有多组截取的图片，所以升维度，转成numpy数组
                                    # print(crop_box)
                                    img = img.crop(crop_box[0])#按坐标截图，按每张图的坐标
                                    img = img.resize((48, 48))#缩放成12*12的
                                    dst_img_filename = str(count) + ".jpg"#命名图片名
                                    dst_img_path = os.path.join(DST_IMAGE_DIR, dst_img_filename)#设置要保存的目录和图片名

                                    img_iou = gen_until.iou(box,crop_box)[0]#[0]是降维度，把二维数组变成常量
                                    if img_iou > 0.65:
                                        img.save(dst_img_path)  # 保存图片
                                    elif img_iou > 0.4 and img_iou < 0.65:
                                        img.save(dst_img_path)  # 保存图片
                                    elif img_iou <0.3:
                                        img.save(dst_img_path)  # 保存图片

                                    # print(crop_box.shape)
                                    # print(box.shape)

                                    # print(img_iou)
                                    #用原来的图片label和截切的图片做IOU
                                    if img_iou > 0.65:#IOU大于0.7的保存成正样本，1
                                        dst_f.write( #获得图片文件名，类别，坐标偏移值，并且写入提前建好为txt文件中
                                        "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n"
                                        .format(dst_img_filename, 1, offset_x1, offset_y1, offset_x2,offset_y2,
                                        offset_lefteye_x,offset_lefteye_y,offset_righteye_x,offset_righteye_y,
                                        offset_nose_x,offset_nose_y,offset_leftmouth_x,offset_leftmouth_y,
                                        offset_rightmouth_x,offset_rightmouth_y))
                                    elif img_iou > 0.4 and img_iou<0.65:#部分样本，2
                                        dst_f.write(  # 获得图片文件名，类别，坐标偏移值，并且写入提前建好为txt文件中
                                        "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n"
                                        .format(dst_img_filename, 2, offset_x1, offset_y1, offset_x2, offset_y2,
                                        offset_lefteye_x, offset_lefteye_y, offset_righteye_x,offset_righteye_y,
                                        offset_nose_x, offset_nose_y, offset_leftmouth_x,offset_leftmouth_y,
                                        offset_rightmouth_x, offset_rightmouth_y))
                                    elif img_iou <0.3:#负样本，0
                                        dst_f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n"
                                        .format(dst_img_filename, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

                                    dst_f.flush()#将正样本，部分样本，负样本的标签打乱保存

                                count += 1#记录图片文件名
                                print(count)

                        else:
                            continue
