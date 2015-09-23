#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <limits>
#include <boost/regex.hpp>

namespace jvl
{
  extern const float inf;
  extern const double qnan;
  extern const float metric_size;
  extern const double heat_map_res;
  extern const double image_res;
  extern const int DEPTH_INTER_STRATEGY;
  
  struct RegExample
  {
    cv::Rect handBB;
    cv::Mat Z;
    std::vector<cv::Vec3d> uvd;
    int index;
    std::string src;
  };

  template<typename T>
  T clamp(T min, T value, T max)
  {
    if(value > max)
      return max;
    if(value < min)
      return min;
    return value;
  }

  RegExample load_icl_datum(const std::string&set_id,int index = -1);
  RegExample load_nyu_datum(const std::string&set_id,int index = -1);
  RegExample load_direc_datum(const std::string&direc);
  int set_size(const std::string&set_id);    

  cv::Vec3d vec_affine(cv::Vec3d, const cv::Mat&affine);
  cv::Mat tileCat(std::vector< cv::Mat > ms, bool number = true);
  cv::Mat eq(cv::Mat&im);
  cv::Mat imroi(const cv::Mat&im,cv::Rect roi);
  cv::Mat imrotate(const cv::Mat&m, double angle_in_radians, cv::Mat&at, cv::Point2f center = cv::Point2f());
  cv::Mat imscale(const cv::Mat&im, double sf, cv::Mat&at);
  cv::Mat imrotate_tight(const cv::Mat&m, double angle_in_radians);
  std::string printfpp(const char* format, ... );
  cv::Mat read_csv_double(const std::string&filename,char separator = ',');
  int readFileToMat(cv::Mat &I, std::string path);
  int writeMatToFile(const cv::Mat &I, std::string path);
  void breakpoint();
  float sample_in_range(float min, float max);
  std::vector<std::string> find_files(std::string dir, boost::regex regex, bool recursive = false);
  cv::Mat horizCat(cv::Mat m1, cv::Mat m2, bool divide = false);
  
  struct Extrema
  {
  public:
    double max;
    double min;
    cv::Point2i maxLoc;
    cv::Point2i minLoc;
  };  
  
  Extrema extrema(const cv::Mat& im);

  double rad2deg(double rad);
  double deg2rad(double deg);  
}
