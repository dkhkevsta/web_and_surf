#ifndef NOMINMAX
#define NOMINMAX
#endif

// std includes
#include <vector>
#include <ctime>

// Qt includes
#include <QApplication>
#include <QDialog>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QLabel>
#include <QTime>
#include <QFile>
#include <QDateTime>
#include <QDebug>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// MySQL includes
#include <my_global.h>
#include <mysql.h>

// Global variables for threads synchronization
QMutex gVideoGuard;
cv::Mat gWebcamImage;
const int kMinIntervalToWriteToDb = 3000; // ms

class WebcamGrabberThread : public QThread {
public:
  WebcamGrabberThread(int web_cam_id, QObject* parent)
      : QThread(parent),
      web_cam_id_(web_cam_id) {
  }

  void Stop(){
    need_stop_ = true;
  }

protected:
  virtual void run() override {
    cv::VideoCapture cap;
    if (!cap.open(web_cam_id_)) {
      assert(!"failed to open webcam");
      return;
    }

    cv::Mat img_webcam;
    while (!need_stop_) {
      cap >> img_webcam;
      QMutexLocker lock(&gVideoGuard);
      gWebcamImage = img_webcam.clone();
    }

    return;
  }

private:
  bool need_stop_ = false;
  const int web_cam_id_ = -1;
};

class WebcamProcessThread : public QThread {
public:
  WebcamProcessThread(int web_cam_id, MYSQL* mysql_object, const std::vector<std::pair<int, std::string>>& patterns, QObject* parent)
    : QThread(parent),
      detector_(400),
      mysql_object_(mysql_object),
      web_cam_id_(web_cam_id) {
    for (const std::pair<int, std::string>& next : patterns) {
      AddImputImage(next.first, next.second);
      AddImputImage(next.first, next.second);
    }
    time_last_image_.restart();
  }

  void AddImputImage(int id, const std::string& file_name) {
    cv::Mat img_match = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (!img_match.data) {
      assert(!"failed to open file");
      return;
    }

    InputImageDescription next_image = { id, img_match };
    detector_.detect(next_image.image, next_image.keypoints);
    extractor_.compute(img_match, next_image.keypoints, next_image.descriptors);
    input_images_.push_back(next_image);
  }

  virtual void run() override {
    while (!need_stop_) {
//      qDebug() << time_last_image_.elapsed();
      if (time_last_image_.elapsed() > kMinIntervalToWriteToDb) {
        cv::Mat img_webcam;
        {
          QMutexLocker lock(&gVideoGuard);
          if (!gWebcamImage.empty()) {
            img_webcam = gWebcamImage.clone();
          }
        }
        if (!img_webcam.empty()) {
          for (size_t nn = 0; nn < input_images_.size(); ++nn) {
            if (IsContains(img_webcam, input_images_.at(nn))) {
              time_last_image_.restart();
              WriteToDb(input_images_.at(nn).id);
              break;
            }
          }
        }
      } else {
        QThread::msleep(100);
      }
    }
    return;
  }

  void WriteToDb(int pattern_id) {
    const std::string query = "INSERT INTO results (cam_id, pattern, time) VALUES ('"
              + std::to_string(web_cam_id_) + "', '"
              + std::to_string(pattern_id) + "', '"
              + QDateTime::currentDateTime().toUTC().toString("yyyy-MM-dd HH:mm:ss").toStdString() + "')";
    if (mysql_query(mysql_object_, query.c_str())) {
      assert(!"failed execute query");
    }
  }

  void Stop() {
    need_stop_ = true;
  }

private:
  struct InputImageDescription {
    int id;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
  };

  bool IsContains(cv::Mat img_scene, const InputImageDescription& input_image) {
    /*
    This code is needed for saving image from web to pass them to recognizer
    cv::imshow("webcam", img_scene);
    if (cvWaitKey(10) == 27) { // ESC key
      cv::imwrite("C:\\cache\\img1.png", img_scene);
    }
    return false;
    */

    std::vector<cv::KeyPoint> keypoints_scene;
    detector_.detect(img_scene, keypoints_scene);
    cv::Mat descriptors_scene;
    extractor_.compute(img_scene, keypoints_scene, descriptors_scene);

    // Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(input_image.descriptors, descriptors_scene, matches);

    if (matches.size() == 0) {
      return false;
    }

    double max_dist = 0;
    double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < input_image.descriptors.rows; i++) {
      double dist = matches[i].distance;
      if (dist < min_dist) {
        min_dist = dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
      }
    }

    // Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    // or a small arbitary value ( 0.02 ) in the event that min_dist is very
    // small)
    // PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;

    for (int i = 0; i < input_image.descriptors.rows; i++) {
      if (matches[i].distance <= std::max(2 * min_dist, 0.02)) {
        good_matches.push_back(matches[i]);
      }
    }

    //-- Draw only "good" matches
    cv::Mat img_matches;
    drawMatches(input_image.image, input_image.keypoints, img_scene, keypoints_scene, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    if (good_matches.size() <= 3) {
      // cv::findHomography accept only count >= 4
      return false;
    }

    //-- Localize the object from img_1 in img_2
    std::vector<cv::Point2f> pts_match;
    std::vector<cv::Point2f> pts_scene;

    for (size_t i = 0; i < good_matches.size(); i++) {
      pts_match.push_back(input_image.keypoints[good_matches[i].queryIdx].pt);
      pts_scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(pts_match, pts_scene, CV_RANSAC);
    CvMat old_h = H;
    // Checking homography
    const bool homography_is_good = CheckHomography(&old_h);

    if (homography_is_good) {
      std::vector<cv::Point2f> corners_match(4);
      corners_match[0] = cv::Point(0, 0);
      corners_match[1] = cv::Point(input_image.image.cols, 0);
      corners_match[2] = cv::Point(input_image.image.cols, input_image.image.rows);
      corners_match[3] = cv::Point(0, input_image.image.rows);

      std::vector<cv::Point2f> corners_scene(4);
      perspectiveTransform(corners_match, corners_scene, H);

      // Draw lines between the corners (the mapped match in the webcam)
      cv::Point2f offset((float)input_image.image.cols, 0.);
      line(img_matches, corners_scene[0] + offset, corners_scene[1] + offset, cv::Scalar(0, 255, 0), 4);
      line(img_matches, corners_scene[1] + offset, corners_scene[2] + offset, cv::Scalar(0, 255, 0), 4);
      line(img_matches, corners_scene[2] + offset, corners_scene[3] + offset, cv::Scalar(0, 255, 0), 4);
      line(img_matches, corners_scene[3] + offset, corners_scene[0] + offset, cv::Scalar(0, 255, 0), 4);
    }

    // for debug purpose
    cv::imshow("result", img_matches);
    cvWaitKey(10);
    return homography_is_good;
  }

  bool CheckHomography(const CvMat* H) {
    const double det = cvmGet(H, 0, 0) * cvmGet(H, 1, 1) - cvmGet(H, 1, 0) * cvmGet(H, 0, 1);
    if (det < 0) {
      return false;
    }

    const double N1 = sqrt(cvmGet(H, 0, 0) * cvmGet(H, 0, 0) + cvmGet(H, 1, 0) * cvmGet(H, 1, 0));
    if (N1 > 4 || N1 < 0.1) {
      return false;
    }

    const double N2 = sqrt(cvmGet(H, 0, 1) * cvmGet(H, 0, 1) + cvmGet(H, 1, 1) * cvmGet(H, 1, 1));
    if (N2 > 4 || N2 < 0.1) {
      return false;
    }

    const double N3 = sqrt(cvmGet(H, 2, 0) * cvmGet(H, 2, 0) + cvmGet(H, 2, 1) * cvmGet(H, 2, 1));
    if (N3 > 0.002) {
      return false;
    }

    return true;
  }

  bool need_stop_ = false;
  cv::SurfFeatureDetector detector_;
  cv::SurfDescriptorExtractor extractor_;
  std::vector<InputImageDescription> input_images_;
  QTime time_last_image_;
  MYSQL* mysql_object_ = nullptr;
  const int web_cam_id_ = 0;
};

int main(int argc, char* argv[]) {
  if (argc != 2) {
    assert(!"invalid usage");
    return 1;
  }

  // opening DB
  MYSQL* mysql_object = mysql_init(nullptr);
  if (!mysql_object) {
    assert(!"failed create db object");
    return 1;
  }

  if (mysql_real_connect(mysql_object, "127.0.0.1", "sploid", "sploid", "recognition", 3306, 0, 0) == nullptr) {
    assert(!"failed connect to db");
    mysql_close(mysql_object);
    mysql_object = nullptr;
    return 1;
  }

  if (mysql_set_character_set(mysql_object, "utf8")) {
    assert(!"failed set UTF8 charset");
    mysql_close(mysql_object);
    mysql_object = nullptr;
    return 1;
  }

  // loading patterns from MySQL
  if (mysql_query(mysql_object, "SELECT id, path FROM patterns")) {
    assert(!"failed execut query");
    mysql_close(mysql_object);
    mysql_object = nullptr;
    return 1;
  }

  MYSQL_RES* result = mysql_store_result(mysql_object);
  if (!result) {
    assert(!"failed store query results");
    mysql_close(mysql_object);
    mysql_object = nullptr;
    return 1;
  }

  MYSQL_ROW row = mysql_fetch_row(result);
  std::vector<std::pair<int, std::string>> patterns;
  while (row) {
    patterns.push_back(std::make_pair(atoi(row[0]), std::string(row[1])));
    row = mysql_fetch_row(result);
  }
  mysql_free_result(result);
  result = nullptr;
  // checking that all files exists
  for (const std::pair<int, std::string>& next : patterns) {
    if (!QFile::exists(next.second.c_str())) {
      assert(!"file not exists");
      return 1;
    }
  }

  QApplication a(argc, argv);

  // starting work
  const int web_cam_id = atoi(argv[1]);
  WebcamGrabberThread grabber_thread(web_cam_id, &a);
  grabber_thread.start();
  WebcamProcessThread work_thread(web_cam_id, mysql_object, patterns, &a);
  work_thread.start();

  // processing GUI events
  const int exec_res = a.exec();

  // stopping work
  grabber_thread.Stop();
  grabber_thread.wait();
  work_thread.Stop();
  work_thread.wait();
  mysql_close(mysql_object);
  mysql_object = nullptr;
  return exec_res;
}