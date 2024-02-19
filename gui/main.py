import sys
from pathlib import Path
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

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


# global variables
app_title = 'USA Medical Cost Prediction'
charges_data = []
predicted_charges = 36000
# taken from the csv file, may be adjusted after reading CSV
min_bmi = 15
max_bmi = 50


# global functions

# save the figures as high-res PNGs 
IMAGES_PATH = Path() / "images" 
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
        global charges_data
        
        wid = QWidget()
        
        layout2 = QVBoxLayout()
       
        self.sc2 = MplCanvas(self, width=5, height=4, dpi=100)
        charges_data.hist(ax=self.sc2.axes)
        save_fig(self.sc2.figure,"histogram", tight_layout=True, fig_extension="png", resolution=300) 

        layout2.addWidget(self.sc2)
        self.setCentralWidget(wid)
        wid.setLayout(layout2)
   

class Window(QMainWindow):
    def __init__(self):
      super().__init__()
      self.load_data()
      self.init_ui()

    def load_data(self):
        global charges_data
        
        self.charges_data = pd.read_csv(os.path.join(os.getcwd(), "data/medical_cost.csv"))
        
        # drop unnecessary column
        X2 = self.charges_data.drop("Id", axis=1)

        # get rid of string with label encoder
        label_encoder = LabelEncoder()
        X2['sex'] = label_encoder.fit_transform(X2['sex'])
        X2['smoker'] = label_encoder.fit_transform(X2['smoker'])
        
        # get rid of string with one hot encoder
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(X2[['region']]).toarray())
        enc_df.rename(columns={3:'region_southwest', 2:'region_southeast', 
                           1:'region_northwest', 0:'region_northeast'},
                           inplace=True)
        
        # drop the region-column (string variant)
        X2 = X2.drop('region', axis=1)

        # join two dataframes
        X2 = enc_df.join(X2)
        
        # assign to global variable
        charges_data = self.charges_data

        # extract the values from a dataframe
        X, y = X2.iloc[:, :-1].values, X2.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)
        
        # Random Forest Regressor
        self.regressor = RandomForestRegressor(n_estimators=5, random_state=0, max_depth=5)
        self.regressor.fit(X_train, y_train)

        Y_pred = self.regressor.predict(X_test)
        self.test_y = y_test       

        # evaluate the model
        mse = mean_squared_error(y_test, Y_pred)
        print(f"mse: {mse}")

        mae = mean_absolute_error(y_test, Y_pred)
        print(f"mae: {mae}")

        r2 = r2_score(y_test, Y_pred)
        print(f"r2: {r2}")


    def show_second_window(self):
        if self.w2.isHidden(): 
            self.w2.show()
    
    def init_ui(self):
      global min_bmi
      global max_bmi
      global predicted_charges
      
      print("init_ui")
      try:
        self.w2 = SecondWindow()
        # set Appilcation default styles
        font = QtGui.QFont("Sanserif", 12)
        # font.setStyleHint(QtGui.QFont.)
        QApplication.setFont(QtGui.QFont(font))
        QApplication.setWindowIcon(QtGui.QIcon('application-document.png'))
        
        # grid layout
        layout = QGridLayout()

        # 1st row of GridLayout
        # dialer
        self.qd = QDial()
        self.qd.setMinimum(18)
        self.qd.setMaximum(64)
        self.qd.setValue(20)
        self.qd.valueChanged.connect(self.updateLabSize)
        layout.addWidget(self.qd, 0, 0)

        # slider
        self.tsl = QSlider(Qt.Orientation.Horizontal)
        self.tsl.setMinimum(min_bmi) 
        self.tsl.setMaximum(max_bmi)
        self.tsl.valueChanged.connect(self.updateSelectedBMI)
        layout.addWidget(self.tsl, 0,1)
        
        # 2nd row of GridLayout
        self.labAppSize = QLabel(self)
        self.labAppSize.setText(" Age : " + str(self.qd.value()))
        layout.addWidget(self.labAppSize, 1,0)
        # in 2nd cell of 2nd row we use a Horizontal Layout for the 2 labels
        layout2 = QHBoxLayout()
        widget2 = QWidget()
        self.minBMI = QLabel(self)
        self.maxBMI = QLabel(self)
        self.selectedBMI = QLabel(self)
        self.selectedBMI.setStyleSheet("QLabel {color: magenta}")
        self.minBMI.setText("Body mass index: " + str(min_bmi) + " (select with Slider)")
        self.maxBMI.setText("           to: " + str(max_bmi))
        self.selectedBMI.setText(str(min_bmi))
        layout2.addWidget(self.minBMI)
        layout2.addWidget(self.selectedBMI)
        layout2.addWidget(self.maxBMI)
        widget2.setLayout(layout2)
        layout.addWidget(widget2, 1, 1)
        
        # 3rd row of GridLayout
        layout3 = QVBoxLayout()
        widget3 = QWidget()

        # slider to choose the number of children
        self.lab_children_num = QLabel(self)
        self.lab_children_num.setText("Number of children (0-10)")
        layout3.addWidget(self.lab_children_num)
        self.slChild = QSlider(Qt.Orientation.Horizontal)
        self.slChild.setMinimum(0) # min number of children
        self.slChild.setMaximum(10) # max number of children (it is 10 and god bless those who have more...)
        self.slChild.valueChanged.connect(self.updateLabPos)
        layout3.addWidget(self.slChild)
        self.labChild = QLabel(self)
        self.labChild.setText("1")
        layout3.addWidget(self.labChild)

        # checkboxes
        # male
        self.cbKitchen = QCheckBox()
        self.cbKitchen.setText("Male")
        layout3.addWidget(self.cbKitchen)

        # female
        self.cbBath = QCheckBox()
        self.cbBath.setText("Female")
        layout3.addWidget(self.cbBath)

        # smoker
        self.cbHeat = QCheckBox()
        self.cbHeat.setText("Smoker")
        layout3.addWidget(self.cbHeat)

        # regions-label
        label = QLabel("Regions:")
        layout3.addWidget(label)

        # southeast
        self.cbSE = QCheckBox()
        self.cbSE.setText("Southeast")
        layout3.addWidget(self.cbSE)

        # southwest
        self.cbSW = QCheckBox()
        self.cbSW.setText("Southwest")
        layout3.addWidget(self.cbSW)

        # northeast
        self.cbNE = QCheckBox()
        self.cbNE.setText("Northeast")
        layout3.addWidget(self.cbNE)

        # northwest
        self.cbNW = QCheckBox()
        self.cbNW.setText("Northwest")
        layout3.addWidget(self.cbNW)
        
        # button to predict costs
        btn = QPushButton("Predict charges", self)
        btn.setToolTip("Show Prediction")
        btn.clicked.connect(self.show_prediction)

        btn.resize(btn.sizeHint())
        # btn.move(410, 118)
        
        # close button set underneath the grid
        layout3.addWidget(btn)

        self.labPrediction = QLabel(self)
        self.labPrediction.setText("Predicted charges: " + str(predicted_charges))
        layout3.addWidget(self.labPrediction)
        
        widget3.setLayout(layout3)
        layout.addWidget(widget3, 2, 0)
        
        # canvas for plot
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_charges(35.3,predicted_charges)
        
        self.sc.setMinimumWidth(100)
        layout.addWidget(self.sc, 2,1)
        
        widget = QWidget()
        widget.setLayout(layout)
        
        # menu

        # first action - Show Historgrams Window
        button_action1 = QAction(QIcon("application-block.png"), "&Histogram", self)
        button_action1.setStatusTip("Show histograms of data")
        button_action1.triggered.connect(self.show_second_window)

        # second action
        button_action2 = QAction(QIcon("store.png"), "&Save Prediction Image", self)
        button_action2.setStatusTip("Save Image")
        button_action2.triggered.connect(self.sClick)
        button_action2.setCheckable(True)

         # third action
        button_action3 = QAction(QIcon("external.png"), "&Close", self)
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
        self.setWindowTitle(app_title)
        self.setGeometry(30, 30, 700, 550)
        self.show()
        return True
      except:
        print("Failed to initialize UI.")
        sys.exit()

    # print bmi (y-axis) and charges (x-axis)
    def plot_charges(self, bmi=None, charges=None):
        global charges_data
        
        self.sc.axes.cla()

        self.df = charges_data.loc[:, ('charges', 'bmi')]
        
        self.df.sort_values(by=['bmi'], inplace=True)
        self.df.reset_index(drop = True, inplace = True)

        self.df.set_index('bmi').plot(ax=self.sc.axes, color="g", linestyle="-")
        
        self.sc.axes.plot(bmi, charges, marker=".", markersize=7, color='m')
        save_fig(self.sc.figure, "prediction_plot", tight_layout=True, fig_extension="png", resolution=300)  # save the figure
        
        self.sc.draw()

    def updateSelectedBMI(self):
        val = self.tsl.value()
        self.selectedBMI.setText(str(val))
        print(val)
        
    def updateLabSize(self):
        val = self.qd.value()
        self.labAppSize.setText(" Age : " + str(val))
        print(val)
    
    def updateLabPos(self):
        val = self.slChild.value()
        self.labChild.setText(str(val))
        print(val)
        
    def sClick(self, event):
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
        
    def show_prediction(self):
        global predicted_charges
        
        a = self.qd.value()  # age
        b = self.tsl.value()  # bmi
        c = self.slChild.value()  # children
        
        # male
        m = 0
        if self.cbKitchen.isChecked:
            m = 1

        # smoker
        h = 0
        if self.cbHeat.isChecked:
            h = 1

        # regions
        # southeast
        se = 0
        if self.cbSE.isChecked:
            se = 1

        #southwest
        sw = 0
        if self.cbSW.isChecked:
            sw = 1

        # northeast
        ne = 0
        if self.cbNE.isChecked:
            ne = 1
        
        # northwest
        nw = 0
        if self.cbNW.isChecked:
            nw = 1
        
        X_test = [[ne, nw, se, sw, a, m, b, c, h]]
        predicted_charges = round(self.regressor.predict(X_test)[0], 2)

        print(type(predicted_charges))
        print("Predicted charges: %.2f" % predicted_charges)

        self.labPrediction.setText("Predicted charges: " + str(predicted_charges))
        self.plot_charges(bmi=b, charges=predicted_charges)


if __name__ == '__main__':
  app = QApplication(sys.argv)
  w = Window()
  sys.exit(app.exec())