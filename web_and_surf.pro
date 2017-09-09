QT += widgets multimedia multimediawidgets

CONFIG += c++11
CONFIG -= flat

SOURCES += main.cpp

# !! CHANGE !! config OpenCV
OPENCV_BUILD_PATH = C:\soft\opencv2413\opencv\build
OPENCV_VERSION = 2413

INCLUDEPATH += $${OPENCV_BUILD_PATH}\include

CONFIG(debug, debug|release) {
  LIB_POSTFIX = 2413d
} else {
  LIB_POSTFIX = 2413
}

LIBS = \
  -L$${OPENCV_BUILD_PATH}\x86\vc12\lib \
  -lopencv_calib3d$${LIB_POSTFIX} \
  -lopencv_flann$${LIB_POSTFIX} \
  -lopencv_highgui$${LIB_POSTFIX} \
  -lopencv_features2d$${LIB_POSTFIX} \
  -lopencv_imgproc$${LIB_POSTFIX} \
  -lopencv_core$${LIB_POSTFIX} \
  -lopencv_nonfree$${LIB_POSTFIX}

# config MySQL
INCLUDEPATH += "C:\Program Files (x86)\MySQL2\MySQL Server 5.7\include"
LIBS += -L"C:\Program Files (x86)\MySQL2\MySQL Server 5.7\lib" -llibmysql
