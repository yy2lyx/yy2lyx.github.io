---
layout: post
title: Keras训练模型部署到C++环境并生成DLL
date: 2020-11-03 00:00:00 +0300
description: Keras训练模型部署到C++环境并生成DLL # Add post description (optional)
img: 1.jpeg # Add image post (optional)
tags: [keras, 部署, C++] # add tag
---


### 一. 准备工作

* 在Windows上搭建完成Tensorflow的C++环境，这里参考本人的上一篇博客
* 在windows上部署opencv3的C++环境
* python3的keras和tf2.X的环境：`pip3 install keras==2.4.3`和`pip3 install tensorflow==2.3.1`
* python3的opencv环境：`pip3 install opencv-python`

### 二. python模型训练

> python模型构建：[Kaggle猫狗分类](https://www.kaggle.com/tongpython/cat-and-dog)
>
> python完整代码放在本人的[git仓库]()上面

#### 2.1 模型训练，保存.h5格式的模型文件

在keras的`model.save()`模型保存的函数中，只支持2种保存方式：

* .h5格式的文件进行保存模型整个结构及其权重。

* 以文件夹（包含assets  saved_model.pb  variables）来保存，模型架构和训练配置（包括优化器、损失和指标）存储在 `saved_model.pb` 中。权重保存在 `variables/` 目录下。

因此使用.h5格式来存储模型结构及权重。

#### 2.2 h5文件转pt文件

将训练好的模型以.h5格式进行存储，再通过转成.pt格式的模型文件，提供C++的tensorflow调用模型。

```python
from keras.models import load_model
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.framework import graph_io
# 针对tf2.x来说不支持freezegraph的，这里需要使用tf1的方式
tf.compat.v1.disable_eager_execution()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


"""----------------------------------配置路径-----------------------------------"""
h5_model_path = 'model/model.h5'
pb_model_name = 'model.pt'
output_path = '.'

"""----------------------------------导入keras模型------------------------------"""
K.set_learning_phase(0)
net_model = load_model(h5_model_path)

print('input is :', net_model.input.name)
print('output is:', net_model.output.name)

"""----------------------------------保存为.pb格式------------------------------"""
sess = K.get_session()
frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in net_model.outputs])
graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)
```

#### 2.3 测试pt模型文件

对同一个图片分别利用.h5模型文件和.pt模型文件进行预测。.pt格式的模型预测代码如下：

```python
def pred_with_pt(img, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.compat.v1.GraphDef()

        # 打开.pb模型
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print("tensors:", tensors)

        # 在一个session中去run一个前向
        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            op = sess.graph.get_operations()

            input_x = sess.graph.get_tensor_by_name("conv2d_input:0")  # 具体名称看上一段代码的input.name
            print("input_X:", input_x)

            out_softmax = sess.graph.get_tensor_by_name("dense/Softmax:0")  # 具体名称看上一段代码的output.name
            print("Output:", out_softmax)

            img_out_softmax = sess.run(out_softmax,
                                       feed_dict={input_x: img})

            return img_out_softmax
```

### 三. C++ 模型预测

#### 3.1 将模型预测类进行封装并生成动态链接库DLL

* VS2019部署（参考本人的上一篇博客）：新建一个动态链接库(DLL)，新建一个头文件`tf_clf.h`和源文件`tf_clf.cpp`，这里是生成DLL包（点击`生成` ==> `重新生成DLLTF`）

* 这里要把`tensorflow_cc.dll`放到生成的`x64/release`里面

* 这里还需要在Release属性页里配置`C/C++`的预处理器(防止后面编译时出现这种错误`tstring.h(350,40): error C2589: “(”:“::”`)：

	```bash
_XKEYCHECK_H
NOMINMAX
	```

* 头文件`tf_clf.h`中声明类的导出

  ```C++
  #pragma once
  #ifndef TF_CLF_H
  #endif // !TF_CLF_H
  
  
  #include <fstream>
  #include <utility>
  #include <Eigen/Core>
  #include <Eigen/Dense>
  #include <iostream>
  
  #include "tensorflow/cc/ops/const_op.h"
  #include "tensorflow/cc/ops/image_ops.h"
  #include "tensorflow/cc/ops/standard_ops.h"
  
  #include "tensorflow/core/framework/graph.pb.h"
  #include "tensorflow/core/framework/tensor.h"
  
  #include "tensorflow/core/graph/default_device.h"
  #include "tensorflow/core/graph/graph_def_builder.h"
  
  #include "tensorflow/core/lib/core/errors.h"
  #include "tensorflow/core/lib/core/stringpiece.h"
  #include "tensorflow/core/lib/core/threadpool.h"
  #include "tensorflow/core/lib/io/path.h"
  #include "tensorflow/core/lib/strings/stringprintf.h"
  
  #include "tensorflow/core/public/session.h"
  #include "tensorflow/core/util/command_line_flags.h"
  
  #include "tensorflow/core/platform/env.h"
  #include "tensorflow/core/platform/init_main.h"
  #include "tensorflow/core/platform/logging.h"
  #include "tensorflow/core/platform/types.h"
  
  #include "opencv2/opencv.hpp"
  
  using namespace tensorflow::ops;
  using namespace tensorflow;
  using namespace std;
  using namespace cv;
  using tensorflow::Flag;
  using tensorflow::Tensor;
  using tensorflow::Status;
  using tensorflow::string;
  using tensorflow::int32;
  
  class __declspec(dllexport) TFClf;
  class TFClf {
  private:
  	vector<float> mean = { 103.939,116.779,123.68 };
  	int resize_col = 224;
  	int resize_row = 224;
  	string input_tensor_name = "conv2d_input";
  	string output_tensor_name = "dense/Softmax";
  	Point draw_point = Point(50, 50);
  
  public:
  	string image_path, model_path;
  	TFClf(string img, string model) :image_path(img), model_path(model) {}
  	void mat_to_tensor(Mat img, Tensor* output_tensor);
  	Mat preprocess_img(Mat img);
  	void model_pred();
  	void show_result_pic(Mat img, int output_class_id, double output_prob);
  };
  ```

* 源文件`tf_clf.cpp`完成类的具体实现，注意这里要`#include "pch.h"`

  ```C++
  #include "pch.h"
  #include "tf_clf.h"
  
  
  Mat TFClf::preprocess_img(Mat img) {
      //图像进行resize处理
      resize(img, img, cv::Size(resize_col, resize_row));
  
      //去均值
      img.convertTo(img, CV_32FC3);
      // 遍历所有的像素的每个通道，减去对应的均值（BGR这种顺序）
      for (int row = 0; row < img.rows; row++) {
          for (int col = 0; col < img.cols; col++) {
              img.at<Vec3f>(row, col)[0] -= mean[0];
              img.at<Vec3f>(row, col)[1] -= mean[1];
              img.at<Vec3f>(row, col)[2] -= mean[2];
          }
      }
      return img;
  }
  
  void TFClf::mat_to_tensor(Mat img, Tensor* output_tensor) {
      //创建一个指向tensor的内容的指针
      float* p = output_tensor->flat<float>().data();
  
      //创建一个Mat，与tensor的指针绑定,改变这个Mat的值，就相当于改变tensor的值
      cv::Mat tempMat(resize_row, resize_col, CV_32FC3, p);
      img.convertTo(tempMat, CV_32FC3);
  }
  
  void TFClf::show_result_pic(Mat img, int output_class_id,double output_prob) {
      string rs_string;
      if (output_class_id == 0) {
          rs_string = "Class:Dog Proba:" + to_string(output_prob);
      }
      else {
          rs_string = "Class:Cat Proba:" + to_string(output_prob);
      }
      // 把label输出到图片中
      cv::putText(img, rs_string, draw_point, FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(255, 200, 200), 2, CV_AA);
      cv::imshow("result", img);
      cv::waitKey(0);
  }
  
  void TFClf::model_pred() {
      /*--------------------------------创建session------------------------------*/
      Session* session;
      Status status = NewSession(SessionOptions(), &session);//创建新会话Session
  
      /*--------------------------------从pb文件中读取模型--------------------------------*/
      GraphDef graphdef; //Graph Definition for current model
  
      Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef); //从pb文件中读取图模型;
      if (!status_load.ok()) {
          std::cout << "ERROR: Loading model failed..." << model_path << std::endl;
          std::cout << status_load.ToString() << "\n";
      }
      Status status_create = session->Create(graphdef); //将模型导入会话Session中;
      if (!status_create.ok()) {
          std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;
      }
      std::cout << "<----Successfully created session and load graph.------->" << endl;
  
      /*---------------------------------载入测试图片-------------------------------------*/
      std::cout << endl << "<------------loading test_image-------------->" << endl;
      Mat img = imread(image_path);
  
  
      if (img.empty())
      {
          std::cout << "can't open the image!!!!!!!" << endl;
      }
      // 图像预处理
      Mat resize_img;
      resize_img = preprocess_img(img);
      //创建一个tensor作为输入网络的接口
      Tensor resized_tensor(DT_FLOAT, TensorShape({ 1,resize_row,resize_col,3 }));
  
      //将Opencv的Mat格式的图片存入tensor
      mat_to_tensor(resize_img, &resized_tensor);
  
      /*-----------------------------------用网络进行测试-----------------------------------------*/
      std::cout << endl << "<-------------Running the model with test_image--------------->" << endl;
      //前向运行，输出结果一定是一个tensor的vector
      vector<tensorflow::Tensor> outputs;
      string output_node = output_tensor_name;
      Status status_run = session->Run({ {input_tensor_name, resized_tensor} }, { output_node }, {}, &outputs);
  
      if (!status_run.ok()) {
          std::cout << "ERROR: RUN failed..." << std::endl;
          std::cout << status_run.ToString() << "\n";
      }
      //把输出值给提取出来
      Tensor t = outputs[0];                   // Fetch the first tensor
      auto tmap = t.tensor<float, 2>();        // Tensor Shape: [batch_size, target_class_num]
      int output_dim = t.shape().dim_size(1);  // Get the target_class_num from 1st dimension
      // Argmax: Get Final Prediction Label and Probability
      int output_class_id = -1;
      double output_prob = 0.0;
      for (int j = 0; j < output_dim; j++)
      {
          std::cout << "Class " << j << " prob:" << tmap(0, j) << "," << std::endl;
          if (tmap(0, j) >= output_prob) {
              output_class_id = j;
              output_prob = tmap(0, j);
          }
      }
      // 展示图片结果
      show_result_pic(img, output_class_id, output_prob);
  
  }
  ```

#### 3.2 新建一个项目测试DLL

  * 新建一个控制台的空项目，并将打包好的`DllTF.dll`和`DllTF.lib`复制到工程中

  * 配置属性管理器：这里需要在`Release | x64`添加之前配置好的`opencv_release.props`和`tf_release.props`。

  * 这里要把`tensorflow_cc.dll`放到生成的`x64/release`里面

  * 新建一个头文件`tf_clf.h`

    ```C++
    #pragma once
    #ifndef TF_CLF_H
    
    #endif // !TF_CLF_H
    
    #pragma comment(lib,"DllTF.lib")
    class __declspec(dllexport) TFClf;
    
    #include <fstream>
    #include <utility>
    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include <iostream>
    
    #include "tensorflow/cc/ops/const_op.h"
    #include "tensorflow/cc/ops/image_ops.h"
    #include "tensorflow/cc/ops/standard_ops.h"
    
    #include "tensorflow/core/framework/graph.pb.h"
    #include "tensorflow/core/framework/tensor.h"
    
    #include "tensorflow/core/graph/default_device.h"
    #include "tensorflow/core/graph/graph_def_builder.h"
    
    #include "tensorflow/core/lib/core/errors.h"
    #include "tensorflow/core/lib/core/stringpiece.h"
    #include "tensorflow/core/lib/core/threadpool.h"
    #include "tensorflow/core/lib/io/path.h"
    #include "tensorflow/core/lib/strings/stringprintf.h"
    
    #include "tensorflow/core/public/session.h"
    #include "tensorflow/core/util/command_line_flags.h"
    
    #include "tensorflow/core/platform/env.h"
    #include "tensorflow/core/platform/init_main.h"
    #include "tensorflow/core/platform/logging.h"
    #include "tensorflow/core/platform/types.h"
    
    #include "opencv2/opencv.hpp"
    
    using namespace tensorflow::ops;
    using namespace tensorflow;
    using namespace std;
    using namespace cv;
    using tensorflow::Flag;
    using tensorflow::Tensor;
    using tensorflow::Status;
    using tensorflow::string;
    using tensorflow::int32;
    
    
    
    class TFClf {
    private:
    	vector<float> mean = { 103.939,116.779,123.68 };
    	int resize_col = 224;
    	int resize_row = 224;
    	string input_tensor_name = "conv2d_input";
    	string output_tensor_name = "dense/Softmax";
    	Point draw_point = Point(50, 50);
    
    public:
    	string image_path, model_path;
    	TFClf(string img, string model) :image_path(img), model_path(model) {}
    	void mat_to_tensor(Mat img, Tensor* output_tensor);
    	Mat preprocess_img(Mat img);
    	void model_pred();
    	void show_result_pic(Mat img, int output_class_id, double output_prob);
    };
    ```

* 新建一个源文件`main.cpp`

  ```C++
  # include "tf_clf.h"
  
  int main() {
  	string model_path = "D:/yeyan/pycharm_project/dogcat/model/model.pt";
  	string img_path = "D:/yeyan/pycharm_project/dogcat/data/test_set/test_set/cats/cat.4001.jpg";
  	TFClf clf = TFClf(img_path, model_path);
  	clf.model_pred();
  }
  ```

  

