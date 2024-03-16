import sys
from pathlib import Path
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
import numpy as np

from joblib import load

import tensorflow as tf

from PyQt6 import QtGui
from PyQt6.QtWidgets import (
    QApplication,
    QWidget, QMessageBox, QCheckBox,
    QDial,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout)

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import QCoreApplication, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
matplotlib.use('Qt5Agg')


# constants
REQUEST_REGION_GENDER = "Please select a gender and a region."

# global variables

APP_TITLE = 'USA Medical Cost Prediction'
CHARGES_DATA = []
PREDICTED_CHARGES = 36000

# taken from the csv file, may be adjusted after reading CSV
MIN_BMI = 15
MAX_BMI = 50


# global functions

# save the figures as high-res PNGs 
IMAGES_PATH = Path() / "supplementary/images" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
  
    if tight_layout:
        fig.tight_layout()

    fig.savefig(path, format=fig_extension, dpi=resolution)


# OOP structure of the app
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)


class SecondWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # use global data
        global CHARGES_DATA
        
        wid = QWidget()
        
        layout2 = QVBoxLayout()
       
        self.sc2 = MplCanvas(self, width=5, height=4, dpi=100)
        CHARGES_DATA.hist(ax=self.sc2.axes)
        save_fig(self.sc2.figure,"histogram", tight_layout=True, fig_extension="png", resolution=300) 

        layout2.addWidget(self.sc2)
        self.setCentralWidget(wid)
        wid.setLayout(layout2)
   

class Window(QMainWindow):
    def __init__(self):
      super().__init__()
      self.load_model_data()
      self.init_ui()
      self.resize(750, 750)

    def load_model_data(self):
        global CHARGES_DATA
        
        self.charges_data = pd.read_csv(os.path.join(os.getcwd(), "data/medical_cost.csv"))
        
        # assign to global variable
        CHARGES_DATA = self.charges_data

        # load the Deep Neural Network model
        self.dnn_model = tf.keras.models.load_model(
            filepath=os.path.join(os.getcwd(), "saved_models/saved_model_dnn/dnn_model.tf")
        )

        # load the Wide & Deep Neural Network model
        self.wdnn_model = tf.keras.models.load_model(
            filepath=os.path.join(os.getcwd(), "saved_models/saved_model_wdnn/wdnn_model.tf")
        )

        # load the Random Forest model
        self.forest_model = load(os.path.join(os.getcwd(), "saved_models/saved_model_forest/forest_model.joblib"))

    def show_second_window(self):
        if self.w2.isHidden(): 
            self.w2.show()
    
    def init_ui(self):
      global MIN_BMI
      global MAX_BMI
      global PREDICTED_CHARGES
      
      print("init_ui")
      try:
        self.w2 = SecondWindow()
        # set Appilcation default styles
        font = QtGui.QFont("Sanserif", 12)
        QApplication.setFont(QtGui.QFont(font))
        QApplication.setWindowIcon(QtGui.QIcon(os.path.join(os.getcwd(), "supplementary/icon/icon.jpg")))
        
        # grid layout
        layout = QGridLayout()

        # 1st row of GridLayout
        # dialer for age
        self.qd = QDial()
        self.qd.setMinimum(18)
        self.qd.setMaximum(64)
        self.qd.setValue(20)
        self.qd.valueChanged.connect(self.update_age)
        layout.addWidget(self.qd, 0, 0)

        # slider for bmi
        self.slider_bmi = QSlider(Qt.Orientation.Horizontal)
        self.slider_bmi.setMinimum(MIN_BMI) 
        self.slider_bmi.setMaximum(MAX_BMI)
        self.slider_bmi.valueChanged.connect(self.update_selected_BMI)
        layout.addWidget(self.slider_bmi, 0, 1)
        
        # 2nd row of GridLayout
        layout_age = QHBoxLayout()
        widget_age = QWidget()
        self.lab_text_age = QLabel(self)
        self.lab_text_age.setText(" Age : ")
        self.lab_selected_age = QLabel(self)
        self.lab_selected_age.setStyleSheet("QLabel {color: blue}")
        self.lab_selected_age.setText(str(self.qd.value()))
        font_text_age = self.lab_text_age.font()
        font_text_age.setBold(True)
        font_text_age.setPointSize(12)
        self.lab_text_age.setFont(font_text_age)
        self.lab_selected_age.setFont(font_text_age)
        layout_age.addWidget(self.lab_text_age)
        layout_age.addWidget(self.lab_selected_age)
        widget_age.setLayout(layout_age)
        layout.addWidget(widget_age, 1, 0)

        # in 2nd cell of 2nd row we use a Horizontal Layout for the 2 labels
        layout2 = QHBoxLayout()
        widget2 = QWidget()
        self.min_bmi = QLabel(self)
        self.max_bmi = QLabel(self)
        self.selected_bmi = QLabel(self)
        self.selected_bmi.setStyleSheet("QLabel {color: blue}")
        self.min_bmi.setText("Body mass index: " + str(MIN_BMI) + " (select with Slider)  ")
        self.max_bmi.setText("           to: " + str(MAX_BMI))
        self.selected_bmi.setText(str(MIN_BMI))
        font_bmi = self.min_bmi.font()
        font_bmi.setBold(True)
        font_bmi.setPointSize(12)
        self.min_bmi.setFont(font_bmi)
        self.max_bmi.setFont(font_bmi)
        self.selected_bmi.setFont(font_bmi)
        layout2.addWidget(self.min_bmi)
        layout2.addWidget(self.selected_bmi)
        layout2.addWidget(self.max_bmi)
        widget2.setLayout(layout2)
        layout.addWidget(widget2, 1, 1)
        
        # 3rd row of GridLayout
        layout3 = QVBoxLayout()
        widget3 = QWidget()

        # slider to choose the number of children and the corresponding label above
        self.lab_children_num = QLabel(self)
        self.lab_children_num.setText("Number of children (0-10)")
        font_lab_children_num = self.lab_children_num.font()
        font_lab_children_num.setBold(True)
        font_lab_children_num.setPointSize(12)
        self.lab_children_num.setFont(font_lab_children_num)
        layout3.addWidget(self.lab_children_num)

        self.slider_children = QSlider(Qt.Orientation.Horizontal)
        self.slider_children.setMinimum(0)  # min number of children
        self.slider_children.setMaximum(10)  # max number of children (it is 10 and god bless those who have more...)
        self.slider_children.valueChanged.connect(self.update_children_number)
        layout3.addWidget(self.slider_children)
        self.label_children = QLabel(self)
        self.label_children.setText("0")
        layout3.addWidget(self.label_children)

        # checkboxes
        # male
        self.cb_male = QCheckBox()
        self.cb_male.setText("Male")
        layout3.addWidget(self.cb_male)

        # female
        self.cb_female = QCheckBox()
        self.cb_female.setText("Female")
        layout3.addWidget(self.cb_female)

        # smoker
        self.cb_smoker = QCheckBox()
        self.cb_smoker.setText("Smoker")
        layout3.addWidget(self.cb_smoker)

        # empty line
        self.empty_label = QLabel(self)
        self.empty_label.setText("")
        layout3.addWidget(self.empty_label)

        # regions-label
        label_regions = QLabel("Regions:")
        font_regions = label_regions.font()
        font_regions.setBold(True)
        font_regions.setPointSize(13)
        label_regions.setFont(font_regions)
        layout3.addWidget(label_regions)

        # southeast
        self.cb_se = QCheckBox()
        self.cb_se.setText("Southeast")
        layout3.addWidget(self.cb_se)

        # southwest
        self.cb_sw = QCheckBox()
        self.cb_sw.setText("Southwest")
        layout3.addWidget(self.cb_sw)

        # northeast
        self.cb_ne = QCheckBox()
        self.cb_ne.setText("Northeast")
        layout3.addWidget(self.cb_ne)

        # northwest
        self.cb_nw = QCheckBox()
        self.cb_nw.setText("Northwest")
        layout3.addWidget(self.cb_nw)
        
        # prediction-label
        label_prediction = QLabel("Predict with:")
        font_prediction = label_prediction.font()
        font_prediction.setBold(True)
        font_prediction.setPointSize(13)
        label_prediction.setFont(font_prediction)
        layout3.addWidget(self.empty_label)  # add an empty line between prediction label and regions checkboxes
        layout3.addWidget(label_prediction)

        # button to predict costs with WDNN
        btn_wdnn = QPushButton("Wide&&Deep Neural Network", self)
        btn_wdnn.setToolTip("Show Prediction wuth a WDNN model.")
        btn_wdnn.clicked.connect(self.show_prediction_wdnn)

        btn_wdnn.resize(btn_wdnn.sizeHint())

        # button to predict costs with DNN
        btn_dnn = QPushButton("Deep Neural Network", self)
        btn_dnn.setToolTip("Show Prediction wuth a DNN model.")
        btn_dnn.clicked.connect(self.show_prediction_dnn)

        btn_dnn.resize(btn_dnn.sizeHint())

        # button to predict costs with Random Forest Regressor
        btn_forest = QPushButton("Random Forest Regression", self)
        btn_forest.setToolTip("Show Prediction wuth a random forest regression model.")
        btn_forest.clicked.connect(self.show_prediction_forest)

        btn_forest.resize(btn_forest.sizeHint())
        
        # close buttons set underneath the grid
        layout3.addWidget(btn_wdnn)
        layout3.addWidget(btn_dnn)
        layout3.addWidget(btn_forest)

        self.label_prediction = QLabel(self)
        self.label_prediction.setText("Predicted charges: " + str(PREDICTED_CHARGES))
        font_lab_prediction = self.label_prediction.font()
        font_lab_prediction.setBold(True)
        font_lab_prediction.setFamily("Tahoma")
        self.label_prediction.setFont(font_lab_prediction)
        layout3.addWidget(self.empty_label)  # add an empty line between predicted charges and buttons
        layout3.addWidget(self.label_prediction)
        
        widget3.setLayout(layout3)
        layout.addWidget(widget3, 2, 0)
        
        # canvas for plot
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_charges(35.3, PREDICTED_CHARGES)
        
        self.sc.setMinimumWidth(100)
        layout.addWidget(self.sc, 2, 1)
        
        widget = QWidget()
        widget.setLayout(layout)
        
        # menu
        # first action - Show Historgrams Window
        button_action1 = QAction(QIcon(os.path.join(os.getcwd(), "supplementary/icon/histogram.png")), "&Histogram", self)
        button_action1.setStatusTip("Show histograms of data")
        button_action1.triggered.connect(self.show_second_window)

        # second action
        button_action2 = QAction(QIcon(os.path.join(os.getcwd(), "supplementary/icon/save.png")), "&Save Prediction Image", self)
        button_action2.setStatusTip("Save Image")
        button_action2.triggered.connect(self.save_click)
        button_action2.setCheckable(True)

        # third action
        button_action3 = QAction(QIcon(os.path.join(os.getcwd(), "supplementary/icon/close.jpg")), "&Close", self)
        button_action3.setStatusTip("Close Application")
        button_action3.triggered.connect(self.closeEvent)
        button_action3.setCheckable(True)

        # menubar
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(button_action1)
        file_menu.addAction(button_action2)
        file_menu.addAction(button_action3)
        
        # set the central widget of the window 
        # widget will expand to take up all the space in the window by default
        self.setCentralWidget(widget)
        self.setWindowTitle(APP_TITLE)
        self.setGeometry(30, 30, 700, 550)
        self.show()
        return True
      except:
        print("Failed to initialize UI.")
        sys.exit()

    # print bmi (y-axis) and charges (x-axis)
    def plot_charges(self, bmi=None, charges=None):
        global CHARGES_DATA
        
        self.sc.axes.cla()

        self.df = CHARGES_DATA.loc[:, ('charges', 'bmi')]
        
        self.df.sort_values(by=['bmi'], inplace=True)
        self.df.reset_index(drop = True, inplace = True)

        self.df.set_index('bmi').plot(ax=self.sc.axes, color="g", linestyle="-")
        
        self.sc.axes.plot(bmi, charges, marker=".", markersize=7, color='m')
        save_fig(self.sc.figure, "prediction_plot", tight_layout=True, fig_extension="png", resolution=300)  # save the figure
        
        self.sc.draw()

    def update_selected_BMI(self):
        val = self.slider_bmi.value()
        self.selected_bmi.setText(str(val))
        print(val)
        
    def update_age(self):
        val = self.qd.value()
        self.lab_selected_age.setText(str(val))
        print(val)
    
    def update_children_number(self):
        val = self.slider_children.value()
        self.label_children.setText(str(val))
        print(val)
        
    def save_click(self, event):
        save_fig(self.sc.figure, "prediction_plot")
        
    def closeEvent(self, event):
        # depict a question after clicking on "X"
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? Any unsaved work will be lost.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)

        if (reply == QMessageBox.StandardButton.Close)  : 
            print("Close Event reply close")
            sys.exit()        
        else:
            if (reply == QMessageBox.StandardButton.Save): 
                save_fig(self.sc.figure, "prediction_plot")
                sys.exit()
            else:
                print("Cancel Closing")
                if not type(event) == bool:
                    event.ignore()  

    # validate checkboxes
    def validate_checkboxes(self):
        # gender should have been chosen
        if not (self.cb_male.isChecked() or self.cb_female.isChecked()):
            return False
        
        # region should have been chosen
        if not (self.cb_se.isChecked() or self.cb_sw.isChecked() or self.cb_ne.isChecked() or self.cb_nw.isChecked()):
            return False
    
        return True

    # show prediction with WDNN  
    def show_prediction_wdnn(self):
        if self.validate_checkboxes():
            global PREDICTED_CHARGES
        
            a = self.qd.value()  # age
            b = self.slider_bmi.value()  # bmi
            c = self.slider_children.value()  # children
            
            # male
            m = 0
            if self.cb_male.isChecked:
                m = 1

            # smoker
            h = 0
            if self.cb_smoker.isChecked:
                h = 1

            # regions
            # southeast
            se = 0
            if self.cb_se.isChecked:
                se = 1

            #southwest
            sw = 0
            if self.cb_sw.isChecked:
                sw = 1

            # northeast
            ne = 0
            if self.cb_ne.isChecked:
                ne = 1
            
            # northwest
            nw = 0
            if self.cb_nw.isChecked:
                nw = 1
            
            X_test = [[ne, nw, se, sw, a, m, b, c, h]]
            PREDICTED_CHARGES = round(self.wdnn_model.predict(X_test)[0][0], 2)  # for wdnn

            print(type(PREDICTED_CHARGES))
            print("Predicted charges: %.2f" % PREDICTED_CHARGES)

            self.label_prediction.setText("Predicted charges: " + str(PREDICTED_CHARGES))
            self.plot_charges(bmi=b, charges=PREDICTED_CHARGES)
        else:
            QMessageBox.warning(self, "Validation Error", REQUEST_REGION_GENDER)

    # show prediction with DNN  
    def show_prediction_dnn(self):
        if self.validate_checkboxes():
            global PREDICTED_CHARGES
        
            a = self.qd.value()  # age
            b = self.slider_bmi.value()  # bmi
            c = self.slider_children.value()  # children
            
            # male
            m = 0
            if self.cb_male.isChecked:
                m = 1

            # smoker
            h = 0
            if self.cb_smoker.isChecked:
                h = 1

            # regions
            # southeast
            se = 0
            if self.cb_se.isChecked:
                se = 1

            #southwest
            sw = 0
            if self.cb_sw.isChecked:
                sw = 1

            # northeast
            ne = 0
            if self.cb_ne.isChecked:
                ne = 1
            
            # northwest
            nw = 0
            if self.cb_nw.isChecked:
                nw = 1
            
            X_test = [[ne, nw, se, sw, a, m, b, c, h]]
            PREDICTED_CHARGES = round(self.dnn_model.predict(X_test)[0][0], 2)  # for dnn

            print(type(PREDICTED_CHARGES))
            print("Predicted charges: %.2f" % PREDICTED_CHARGES)

            self.label_prediction.setText("Predicted charges: " + str(PREDICTED_CHARGES))
            self.plot_charges(bmi=b, charges=PREDICTED_CHARGES)
        else:
            QMessageBox.warning(self, "Validation Error", REQUEST_REGION_GENDER)

    # show prediction with Random Forest
    def show_prediction_forest(self):
        if self.validate_checkboxes():
            global PREDICTED_CHARGES
        
            a = self.qd.value()  # age
            b = self.slider_bmi.value()  # bmi
            c = self.slider_children.value()  # children
            
            # male
            m = 0
            if self.cb_male.isChecked:
                m = 1

            # smoker
            h = 0
            if self.cb_smoker.isChecked:
                h = 1

            # regions
            # southeast
            se = 0
            if self.cb_se.isChecked:
                se = 1

            #southwest
            sw = 0
            if self.cb_sw.isChecked:
                sw = 1

            # northeast
            ne = 0
            if self.cb_ne.isChecked:
                ne = 1
            
            # northwest
            nw = 0
            if self.cb_nw.isChecked:
                nw = 1
            
            X_test = [[ne, nw, se, sw, a, m, b, c, h]]
            PREDICTED_CHARGES = round(self.forest_model.predict(X_test)[0], 2)  # for random forest

            print(type(PREDICTED_CHARGES))
            print("Predicted charges: %.2f" % PREDICTED_CHARGES)

            self.label_prediction.setText("Predicted charges: " + str(PREDICTED_CHARGES))
            self.plot_charges(bmi=b, charges=PREDICTED_CHARGES)
        else:
            QMessageBox.warning(self, "Validation Error", REQUEST_REGION_GENDER)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  w = Window()
  sys.exit(app.exec())