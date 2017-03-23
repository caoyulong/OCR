# OCR 手写数字识别
- 可以识别手写数字0-9
- 可在网页中训练和测试
- 通过单隐藏层BP神经网络实现

运行方式：
- 编译运行server.py，然后打开ocr.html进行训练或测试即可
- 需要python2.7，numpy库。

备注：如果有nn.json（存储训练好的神经网络）文件，则初始化神经网络时会自动导入该数据，否则会使用mydata.csv,mydataLabels.csv进行训练,训练耗时约20秒。
参考：《500 lines or less》