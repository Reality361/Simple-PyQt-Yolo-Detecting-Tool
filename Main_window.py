import multiprocessing
from PyQt6 import QtWidgets, QtCore, QtGui
from threading import Thread
import cv2, os, time
import sys

# Disable YOLO verbose logging
os.environ[
    'YOLO_VERBOSE'] = 'False'  # This must be written before importing ultralytics, otherwise it won't take effect
from ultralytics import YOLOv10
from ultralytics import YOLO


class Main_Window(QtWidgets.QMainWindow):

    def __init__(self):
        # Pyinstaller fix
        multiprocessing.freeze_support()
        super().__init__()

        self.setupUI()

        # Set global variables to fix bugs
        self.flag_camera = False
        self.flag_video = False
        self.flag_image = False

        # Default mode is object detection
        self.flag_mode_det = True

        # Default model scale is "n" (nano)
        self.model_scale = 0

        self.bottomLayout.addLayout(self.btnLayout)
        # Set up timers
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        self.timer_video = QtCore.QTimer()
        self.timer_video.timeout.connect(self.show_video)

        # Load model
        self.model = YOLOv10("model\\yolov10n.pt")

        # List to hold frames for analysis
        self.frameToanalyze_camera = []
        self.frameToanalyze_video = []

        # Start frame processing in separate threads
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()
        Thread(target=self.frameAnalyzeThreadFunc_video, daemon=True).start()

        # Detect cameras and update the combo box
        self.detect_and_update_cameras()


    def setupUI(self):
        # Size, icon, and window title
        self.resize(1180, 600)
        self.setWindowTitle("Simple PyQt Yolo Detecting Tool")
        self.setWindowIcon(QtGui.QIcon("images\\icon.png"))

        # Central Widget
        self.centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralWidget)

        mainLayout = QtWidgets.QVBoxLayout(self.centralWidget)

        # Display section (top part)
        self.topLayout = QtWidgets.QHBoxLayout()
        self.label_ori = QtWidgets.QLabel(self)
        self.label_result = QtWidgets.QLabel(self)

        self.label_ori.setMinimumSize(560, 420)
        self.label_result.setMinimumSize(560, 420)
        self.label_ori.setStyleSheet("border:1px solid black")
        self.label_result.setStyleSheet("border:1px solid black")

        self.topLayout.addWidget(self.label_ori)
        self.topLayout.addWidget(self.label_result)

        mainLayout.addLayout(self.topLayout)

        # Control Section (Bottom part)
        self.setup_control_panel(mainLayout)

    def setup_control_panel(self, mainLayout):
        # Control section (bottom part)
        groupBox_left = QtWidgets.QGroupBox(self)
        groupBox_right = QtWidgets.QGroupBox(self)

        self.bottomLayout_control = QtWidgets.QHBoxLayout(groupBox_left)
        self.bottomLayout = QtWidgets.QVBoxLayout(groupBox_right)

        # Create control panel
        # Model size selection
        self.white_add1 = QtWidgets.QLabel(self)
        self.white_add2 = QtWidgets.QLabel(self)
        self.white_add3 = QtWidgets.QLabel(self)

        self.model_label = self.create_label("Select Model:", 20)
        self.model_comba = self.create_combobox(["yolov10n.pt", "yolov10s.pt"], 15)
        self.model_comba.currentIndexChanged.connect(self.on_model_changed)
        self.bottomLayout_control.addWidget(self.model_label)
        self.bottomLayout_control.addWidget(self.model_comba)

        self.bottomLayout_control.addWidget(self.white_add1)

        # Mode selection
        self.mode_label = self.create_label("Select Mode:", 20)
        self.mode_comba = self.create_combobox(["Object Detection", "Object Segmentation"], 15)
        self.mode_comba.currentIndexChanged.connect(self.on_mode_changed)
        self.bottomLayout_control.addWidget(self.mode_label)
        self.bottomLayout_control.addWidget(self.mode_comba)

        self.bottomLayout_control.addWidget(self.white_add2)

        # Camera selection
        self.camera_label = self.create_label("Select Camera:", 20)
        self.camera_comba = self.create_combobox([], 15)
        self.camera_comba.currentIndexChanged.connect(self.change_camera)
        self.bottomLayout_control.addWidget(self.camera_label)
        self.bottomLayout_control.addWidget(self.camera_comba)

        self.bottomLayout_control.addWidget(self.white_add3)

        mainLayout.addWidget(groupBox_left)

        # Control buttons
        self.btnLayout = QtWidgets.QHBoxLayout()
        self.setup_buttons()

        self.bottomLayout.setSpacing(40)
        mainLayout.addWidget(groupBox_right)


    def create_label(self, text, font_size):
        label = QtWidgets.QLabel(self)
        label.setText(text)
        label.setStyleSheet(f"font-size:{font_size}px")
        return label

    def create_combobox(self, items, font_size):
        combo_box = QtWidgets.QComboBox(self)
        combo_box.addItems(items)
        combo_box.setStyleSheet(f"font-size:{font_size}px")
        return combo_box

    def setup_buttons(self):

        button_style = '''
        QPushButton {
            align-items: center;
            appearance: none;
            background-color: #fff;
            border-radius: 24px;
            border-style: none;
            box-shadow: rgba(0, 0, 0, .2) 0 3px 5px -1px,rgba(0, 0, 0, .14) 0 6px 10px 0,rgba(0, 0, 0, .12) 0 1px 18px 0;
            box-sizing: border-box;
            color: #3c4043;
            cursor: pointer;
            display: inline-flex;
            fill: currentcolor;
            font-family: "Google Sans",Roboto,Arial,sans-serif;
            font-size: 20px;
            font-weight: 500;
            height: 48px;
            justify-content: center;
            letter-spacing: .25px;
            line-height: normal;
            overflow: visible;
            padding: 2px 24px;
            position: relative;
            text-align: center;
            text-transform: none;
            transition: box-shadow 280ms cubic-bezier(.4, 0, .2, 1),opacity 15ms linear 30ms,transform 270ms cubic-bezier(0, 0, .2, 1) 0ms;
            user-select: none;
            -webkit-user-select: none;
            touch-action: manipulation;
            width: auto;
            will-change: transform,opacity;
            z-index: 0;
            }

            QPushButton:hover {
            background: #F6F9FE;
            color: #174ea6;
            }

            QPushButton:active {
            box-shadow: 0 4px 4px 0 rgb(60 64 67 / 30%), 0 8px 12px 6px rgb(60 64 67 / 15%);
            outline: none;
            }

            QPushButton:focus {
            outline: none;
            border: 2px solid #4285f4;
            }

            QPushButton:not(:disabled) {
            box-shadow: rgba(60, 64, 67, .3) 0 1px 3px 0, rgba(60, 64, 67, .15) 0 4px 8px 3px;
            }

            QPushButton:not(:disabled):hover {
            box-shadow: rgba(60, 64, 67, .3) 0 2px 3px 0, rgba(60, 64, 67, .15) 0 6px 10px 4px;
            }

            QPushButton:not(:disabled):focus {
            box-shadow: rgba(60, 64, 67, .3) 0 1px 3px 0, rgba(60, 64, 67, .15) 0 4px 8px 3px;
            }

            QPushButton:not(:disabled):active {
            box-shadow: rgba(60, 64, 67, .3) 0 4px 4px 0, rgba(60, 64, 67, .15) 0 8px 12px 6px;
            }

            QPushButton:disabled {
            box-shadow: rgba(60, 64, 67, .3) 0 1px 3px 0, rgba(60, 64, 67, .15) 0 4px 8px 3px;
            }
        '''

        self.cam_btn = self.create_button("Camera📹", self.startCamera, button_style)
        self.video_btn = self.create_button("Video📽️", self.getVideo, button_style)
        self.img_btn = self.create_button("Image🏞️", self.get_image, button_style)
        self.stop_btn = self.create_button("Stop⏸️", self.stop, button_style)

        self.btnLayout.addWidget(self.cam_btn)
        self.btnLayout.addWidget(self.video_btn)
        self.btnLayout.addWidget(self.img_btn)
        self.btnLayout.addWidget(self.stop_btn)

        self.bottomLayout.setSpacing(40)

    def create_button(self, text, func, button_style):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(func)
        button.setStyleSheet(button_style)
        return button

    def detect_and_update_cameras(self):
        # Check available cameras
        self.camera_list = []
        for i in range(10):  # Check up to 10 possible cameras
            #cap = cv2.VideoCapture(i)
            self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.camera_list.append(f"Camera {i}")
                self.cap.release()

        self.camera_comba.clear()
        self.camera_comba.addItems(self.camera_list)

    def ChangeFlag_camera(self):
        self.flag_camera = True

    def getVideo(self):
        file_dialog = QtWidgets.QFileDialog()
        self.video_path = file_dialog.getOpenFileName(
            self,
            "Select the video to upload",
            "C:\\",
            "Video Types(*.avi *.mp4 *.mov *.flv)"
        )[0]
        if self.video_path == "":
            return
        self.stop()
        self.flag_video = True
        self.cap_video = cv2.VideoCapture(self.video_path)
        if not self.cap_video.isOpened():
            print("Failed to open video, please try again")
            return
        if self.timer_video.isActive() == False:
            self.timer_video.start(27)

    def show_video(self):
        ret, frame = self.cap_video.read()

        if not ret:
            return

        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format.Format_RGB888)

        self.label_ori.setPixmap(QtGui.QPixmap.fromImage(qImage))

        if not self.frameToanalyze_video:
            self.frameToanalyze_video.append(frame)

    def startCamera(self):
        self.stop()
        # Open camera
        camera_index = self.camera_comba.currentIndex()
        self.cap_camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap_camera.isOpened():
            print("Failed to open camera, please check the device")
            return
        if self.timer_camera.isActive() == False:
            self.timer_camera.start(27)

    def show_camera(self):
        ret, frame = self.cap_camera.read()

        if not ret:
            return

        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format.Format_RGB888)

        self.label_ori.setPixmap(QtGui.QPixmap.fromImage(qImage))

        if not self.frameToanalyze_camera:
            self.frameToanalyze_camera.append(frame)

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToanalyze_camera:
                time.sleep(0.01)
                continue

            frame = self.frameToanalyze_camera.pop(0)

            results = self.model(frame)[0]

            if self.flag_mode_det:
                img = results.plot(line_width=1)
            else:
                img = results.plot(line_width=0)

            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format.Format_RGB888)

            self.label_result.setPixmap(QtGui.QPixmap.fromImage(qImage))

            time.sleep(0.02)

    def frameAnalyzeThreadFunc_video(self):
        while True:
            if not self.frameToanalyze_video:
                time.sleep(0.01)
                continue

            frame = self.frameToanalyze_video.pop(0)

            results = self.model(frame)[0]

            if self.flag_mode_det:
                img = results.plot(line_width=1)
            else:
                img = results.plot(line_width=0)
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0],
                                  QtGui.QImage.Format.Format_RGB888)

            self.label_result.setPixmap(QtGui.QPixmap.fromImage(qImage))

            time.sleep(0.02)

    def on_model_changed(self, index):
        detection_paths = ["model\\yolov10n.pt", "model\\yolov10s.pt"]
        segmentation_paths = ["model\\yolov8n-seg.pt", "model\\yolov8s-seg.pt"]
        if self.flag_mode_det:
            model_path = detection_paths[index]
            self.model = YOLOv10(model_path)
        else:
            model_path = segmentation_paths[index]
            self.model = YOLO(model_path)
        if self.flag_image:
            self.show_image()

    def on_mode_changed(self, index):
        detection_paths = ["model\\yolov10n.pt", "model\\yolov10s.pt"]
        segmentation_paths = ["model\\yolov8n-seg.pt", "model\\yolov8s-seg.pt"]
        if index == 0:
            self.flag_mode_det = True
            mode_path = detection_paths[self.model_scale]
            self.model = YOLOv10(mode_path)
        elif index == 1:
            self.flag_mode_det = False
            mode_path = segmentation_paths[self.model_scale]
            self.model = YOLO(mode_path)
        if self.flag_image:
            self.show_image()

    def stop(self):
        if self.timer_video.isActive():
            self.timer_video.stop()
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if self.flag_video:
            self.cap_video.release()
        if self.flag_camera:
            self.cap_camera.release()

    def get_image(self):
        file_dialog = QtWidgets.QFileDialog()
        self.image_path = file_dialog.getOpenFileName(
            self,
            "Select the image to upload",
            "D:\\",
            "Image Files(*.jpg *.jpeg *.png *.bmp)"
        )[0]
        if self.image_path == "":
            return
        self.stop()
        self.flag_image = True
        frame = cv2.imread(self.image_path)
        self.flag_image = False
        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format.Format_RGB888)
        self.label_ori.setPixmap(QtGui.QPixmap.fromImage(qImage))
        self.frameToanalyze_camera.append(frame)

    def change_camera(self):
        # Stop the current camera and start the new one when the user selects a new camera
        if self.flag_camera:  # If a camera is already running, stop it
            self.stop()

        # Get the selected camera index
        self.startCamera()  # Start the new camera

    def closeEvent(self, event: QtCore.QEvent):
        self.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Main_Window()
    window.show()
    sys.exit(app.exec())
