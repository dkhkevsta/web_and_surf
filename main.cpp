// std includes
#include <vector>

// Qt includes
#include <QApplication>
#include <QDialog>
#include <QMediaPlayer>
#include <QMediaPlaylist>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>
#include <QVideoWidget>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// Global variables for threads synchronization
QMutex gVideoGuard;
int gVideoIndex = -1;

class WebcamProcessThread : public QThread {
public:
  WebcamProcessThread(QObject* parent)
    : QThread(parent),
      detector_(400) {

    // !! CHANGE !! set image paths
    AddImputImage("C:\\cache\\1.png");
    AddImputImage("C:\\cache\\2.png");
  }

  void AddImputImage(const std::string& file_name) {
    cv::Mat img_match = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
    if (!img_match.data) {
      assert(!"failed to open file");
      return;
    }

    InputImageDescription next_image = { img_match };
    detector_.detect(next_image.image, next_image.keypoints);
    extractor_.compute(img_match, next_image.keypoints, next_image.descriptors);
    input_images_.push_back(next_image);
  }

  virtual void run() override {
    cv::VideoCapture cap;
    if (!cap.open(0)) {
      assert(!"failed to open webcam");
      return;
    }

    cv::Mat img_webcam;
    while (!need_stop_) {
      cap >> img_webcam;
      for (size_t nn = 0; nn < input_images_.size(); ++nn) {
        if (IsContains(img_webcam, input_images_.at(nn))) {
          QMutexLocker lock(&gVideoGuard);
          if (gVideoIndex == -1) {
            gVideoIndex = static_cast<int>(nn);
          }
        }
      }
    }

    return;
  }

  void Stop() {
    need_stop_ = true;
  }

private:
  struct InputImageDescription {
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
    // cv::imshow("result", img_matches);
    // cvWaitKey(10);
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
};

class VideoDialog : public QDialog {
public:
  VideoDialog(QWidget* parent)
    : QDialog(parent) {
    QVBoxLayout* vl = new QVBoxLayout(this);
    player_ = new QMediaPlayer(this);
    connect(player_, &QMediaPlayer::stateChanged, this, &VideoDialog::OnStateChanged);
    QVideoWidget* video_widget = new QVideoWidget(this);
    vl->addWidget(video_widget);
    player_->setVideoOutput(video_widget);
    resize(600, 400);

    QTimer* timer = new QTimer(this);
    timer->setInterval(500);
    connect(timer, &QTimer::timeout, this, &VideoDialog::OnTimerCheckVideoToShow);
    timer->start();

    work_thread_ = new WebcamProcessThread(this);
    work_thread_->start();
  }

  ~VideoDialog() override {
    work_thread_->Stop();
    work_thread_->wait();
  }

private:
  void OnTimerCheckVideoToShow() {
    QMutexLocker lock(&gVideoGuard);
    if (gVideoIndex == -1) {
      return;
    }

    if (player_->state() != QMediaPlayer::StoppedState) {
      return;
    }

    // !! CHANGE !! set video paths
    switch (gVideoIndex) {
    default:
      assert(!"invalid video index");
    case 0:
      player_->setMedia(QUrl::fromLocalFile("C:/cache/sirf/video.3gp"));
      break;
    case 1:
      player_->setMedia(QUrl::fromLocalFile("C:/cache/sirf/video1.3gp"));
      break;
    }
    player_->play();
  }

  void OnStateChanged(QMediaPlayer::State state) {
    if (state == QMediaPlayer::StoppedState) {
      // reset searching on video finish
      QMutexLocker lock(&gVideoGuard);
      gVideoIndex = -1;
    }
  }

  QMediaPlayer* player_ = nullptr;
  WebcamProcessThread* work_thread_ = nullptr;

};

int main(int argc, char* argv[]) {
  QApplication a(argc, argv);
  VideoDialog vd(nullptr);
  vd.show();
  return a.exec();
}