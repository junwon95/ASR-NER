import yaml
import ASR
import Gmasking
from ASR.Ginference import inference
from NER.Interactive_shell_NER import NER
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import Qt
from playsound import playsound
import sys

form_class = uic.loadUiType("./ui/home.ui")[0]
form_class2 = uic.loadUiType("./ui/output.ui")[0]
form_class3 = uic.loadUiType("./ui/masking.ui")[0]
filepath = 'none'

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.fileButton.clicked.connect(self.fileopen)
        self.recognitionButton.clicked.connect(self.openOutputClass)
        self.maskingButton.clicked.connect(self.openMaskingClass)

    def fileopen(self):

        global filepath
        filepath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')[0]
        ASR.Ginference.filename = filepath
        Gmasking.filename = filepath
        # 음성 파일 경로들이 저장되어있는 TEST폴더의 audio_paths.txt를 읽어와서 그 경로를 ASR inference의
        # with open('TEST/audio_paths.txt') as f: 이 부분에 넣음

    def openOutputClass(self):
        if filepath != 'none':
            self.ASR()
            NER()
            openOutputClass = OutputClass()
            openOutputClass.showModal()
        else:
            QMessageBox.about(self, 'Warning', '파일을 선택하지 않았습니다.')

    def openMaskingClass(self):
        if filepath != 'none':
            Gmasking.masking()
            openMaskingClass = MaskingClass()
            openMaskingClass.showModal()
        else:
            QMessageBox.about(self, 'Warning', '파일을 선택하지 않았습니다.')

    def ASR(self):
        with open('ASR/data/config.yaml') as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)

        opt['use_val_data'] = False
        opt['inference'] = True
        opt['eval'] = True
        inference(opt)

class OutputClass(QDialog, form_class2):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.rejectButton.clicked.connect(self.cancel_clicked)
        f = open("./OUTPUTS/NER-OUT/final_output.txt", 'rt', encoding="UTF8")
        self.text = f.read()
        self.final_text.setText(self.text)
        self.final_text.setAlignment(Qt.AlignCenter)
        f.close()
        f = open("./OUTPUTS/ASR-OUT/transcripts.txt", 'rt', encoding="cp949")
        self.text = f.read()
        self.before_text.setText(self.text)
        self.before_text.setAlignment(Qt.AlignCenter)
        f.close()

    def showModal(self):
        return super().exec_()

    def cancel_clicked(self):
        self.reject()

class MaskingClass(QDialog, form_class3):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        f = open("./OUTPUTS/NER-OUT/final_output.txt", 'rt', encoding="UTF8")
        self.m_text = f.read()
        self.resultText.setText(self.m_text)
        self.resultText.setAlignment(Qt.AlignCenter)
        self.playButton.clicked.connect(self.play)
        self.rejectButton.clicked.connect(self.cancel_clicked)

    def showModal(self):
        return super().exec_()

    def play(self):
        playsound('C:/Users/junwonseo95/Desktop/ASR-NER_pipeline/OUTPUTS/audio1.wav')

    def cancel_clicked(self):
        self.reject()

if __name__ == '__main__':
    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    # 프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()
