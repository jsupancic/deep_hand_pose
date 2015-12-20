#include <jvl/jvl.hpp>
#include <mutex>
#include <fstream>
#include <stdarg.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "boost/filesystem.hpp"
#include "boost/algorithm/string/trim.hpp"
#include <jvl/blob_io.hpp>

using namespace std;
using namespace cv;

namespace jvl
{
  const float inf = std::numeric_limits<float>::infinity();
  const double qnan = std::numeric_limits<double>::quiet_NaN();
  const float metric_size = 38;
  const double heat_map_res = 18;
  //const double image_res = 86; // Tompson
  const double image_res = 128; // Oberwegert
  // cv::INTER_NEAREST or cv::INTER_AREA
  const int DEPTH_INTER_STRATEGY = cv::INTER_AREA;
  
  int writeMatToFile(const Mat &I, string path) {
 
    //load the matrix size
    int matWidth = I.size().width, matHeight = I.size().height;
 
    //read type from Mat
    int type = I.type();
 
    //declare values to be written
    float fvalue;
    double dvalue;
    Vec3f vfvalue;
    Vec3d vdvalue;
 
    //create the file stream
    ofstream file(path.c_str(), ios::out | ios::binary );
    if (!file)
      return -1;
 
    //write type and size of the matrix first
    file.write((const char*) &type, sizeof(type));
    file.write((const char*) &matWidth, sizeof(matWidth));
    file.write((const char*) &matHeight, sizeof(matHeight));
 
    //write data depending on the image's type
    switch (type)
    {
    default:
      cout << "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3" << endl;
      break;
      // FLOAT ONE CHANNEL
    case CV_32F:
      //cout << "Writing CV_32F image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	fvalue = I.at<float>(i);
	file.write((const char*) &fvalue, sizeof(fvalue));
      }
      break;
      // DOUBLE ONE CHANNEL
    case CV_64F:
      //cout << "Writing CV_64F image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	dvalue = I.at<double>(i);
	file.write((const char*) &dvalue, sizeof(dvalue));
      }
      break;
 
      // FLOAT THREE CHANNELS
    case CV_32FC3:
      //cout << "Writing CV_32FC3 image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	vfvalue = I.at<Vec3f>(i);
	file.write((const char*) &vfvalue, sizeof(vfvalue));
      }
      break;
 
      // DOUBLE THREE CHANNELS
    case CV_64FC3:
      //cout << "Writing CV_64FC3 image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	vdvalue = I.at<Vec3d>(i);
	file.write((const char*) &vdvalue, sizeof(vdvalue));
      }
      break;
 
    }
 
    //close file
    file.close();
 
    return 0;
  }
 
  int readFileToMat(Mat &I, string path) {
 
    //declare image parameters
    int matWidth, matHeight, type;
 
    //declare values to be written
    float fvalue;
    double dvalue;
    Vec3f vfvalue;
    Vec3d vdvalue;
 
    //create the file stream
    ifstream file(path.c_str(), ios::in | ios::binary );
    if (!file)
      return -1;
 
    //read type and size of the matrix first
    file.read((char*) &type, sizeof(type));
    file.read((char*) &matWidth, sizeof(matWidth));
    file.read((char*) &matHeight, sizeof(matHeight));
 
    //change Mat type
    I = Mat::zeros(matHeight, matWidth, type);
 
    //write data depending on the image's type
    switch (type)
    {
    default:
      cout << "Error: wrong Mat type: must be CV_32F, CV_64F, CV_32FC3 or CV_64FC3" << endl;
      break;
      // FLOAT ONE CHANNEL
    case CV_32F:
      //cout << "Reading CV_32F image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	file.read((char*) &fvalue, sizeof(fvalue));
	I.at<float>(i) = fvalue;
      }
      break;
      // DOUBLE ONE CHANNEL
    case CV_64F:
      //cout << "Reading CV_64F image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	file.read((char*) &dvalue, sizeof(dvalue));
	I.at<double>(i) = dvalue;
      }
      break;
 
      // FLOAT THREE CHANNELS
    case CV_32FC3:
      //cout << "Reading CV_32FC3 image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	file.read((char*) &vfvalue, sizeof(vfvalue));
	I.at<Vec3f>(i) = vfvalue;
      }
      break;
 
      // DOUBLE THREE CHANNELS
    case CV_64FC3:
      //cout << "Reading CV_64FC3 image" << endl;
      for (int i=0; i < matWidth*matHeight; ++i) {
	file.read((char*) &vdvalue, sizeof(vdvalue));
	I.at<Vec3d>(i) = vdvalue;
      }
      break;
 
    }
 
    //close file
    file.close();
 
    return 0;
  }
    
  static Mat imageeq_realize(
    cv::Mat_< float > im,vector<float>&values)
  {
    Mat showMe(im.rows,im.cols,DataType<Vec3b>::type,Scalar(255,255,255));
    if(showMe.size().area() == 0)
      return showMe;
    if(values.size() < 2)
      return showMe;
    
    // discover the quantiles
    std::multimap<float,int> quantiles; // should containe 256 - 1 values
    for(int qIter = 1; qIter < 256; qIter++)
    {
      float quantile = static_cast<float>(qIter)/256;    
      int high_idx = clamp<int>(0,quantile*(values.size() - 1),values.size()-1);
      float threshold = values[high_idx];
      quantiles.insert(pair<float,int>(threshold,qIter));	
    }    
    assert(quantiles.size() == 255);
    
    // write the image
    //printf("q = %f low = %f high = %f\n",quantile,thresh_low,thresh_high);
    for(int rIter = 0; rIter < im.rows; rIter++)
      for(int cIter = 0; cIter < im.cols; cIter++)
      {	
	float curValue = im.at<float>(rIter,cIter);	  
	
	if(std::isnan(curValue))
	  // NAN => Red
	  showMe.at<Vec3b>(rIter,cIter) = Vec3b(0,0,255);
	else if(curValue == inf)
	  // INF => Blue
	  showMe.at<Vec3b>(rIter,cIter) = Vec3b(255,0,0);
	else if(curValue == -inf)
	  // -INF => Green
	  showMe.at<Vec3b>(rIter,cIter) = Vec3b(0,255,0);
	else
	{
	  auto quantile = quantiles.lower_bound(curValue); // equiv or after object
	  int qIter;
	  if(quantile == quantiles.end())
	    qIter = 255;
	  else
	    qIter = quantile->second - 1;
	  showMe.at<Vec3b>(rIter,cIter) = Vec3b(qIter,qIter,qIter);
	}
      }    
    
    return showMe;
  }
  
  Mat horizCat(Mat m1, Mat m2, bool divide)
  { 
    if(m1.empty())
      return m2;
    if(m2.empty())
      return m1;
    
    if(!divide)
    {
      assert(m1.type() == m2.type());
      Mat result(std::max(m1.rows,m2.rows),m1.cols+m2.cols,m1.type(),Scalar::all(0));
      Mat roi1 = result(Rect(Point(0,0),m1.size()));
      Mat roi2 = result(Rect(Point(m1.cols,0),m2.size()));
      m1.copyTo(roi1);
      m2.copyTo(roi2);
      
      return result;
    }
    else
    {
      Mat D(std::max(m1.rows,m2.rows),5,m1.type(),Scalar(127,180,255));
      return horizCat(m1,horizCat(D,m2,false),false);
    }
  }
  
  Mat vertCat(Mat m1, Mat m2, bool divider = false)
  {
    if(m1.empty())
      return m2;
    if(m2.empty())
      return m1;    
    
    assert(m1.type() == m2.type());
    // add a boarder
    if(divider)
    {
      m1 = m1.clone();
      m1 = vertCat(m1,Mat(5,m1.cols,m1.type(),Scalar(127,180,255)),false);
    }
    
    // concat
    assert(m1.type() == m2.type());
    Mat result(m1.rows+m2.rows,std::max(m1.cols,m2.cols),m1.type(),Scalar::all(0));
    Mat roi1 = result(Rect(Point(0,0),m1.size()));
    Mat roi2 = result(Rect(Point(0,m1.rows),m2.size()));
    m1.copyTo(roi1);
    m2.copyTo(roi2);
    
    return result;
  }
  
  bool goodNumber(float n)
  {
    if(n == numeric_limits<float>::infinity())
      return false;
    if(n == -numeric_limits<float>::infinity())
      return false;
    if(n != n)
      return false;
    if(isnan(n))
      return false;
    return true;
  }  
  
  Mat image_text(const Mat& bg, string text, Vec3b color)
  {
    // define our font information
    int font_face = FONT_HERSHEY_PLAIN ;
    double font_scale = 1;
    Scalar font_color(color[0],color[1],color[2]);
    int font_thickness = 1;
    
    // figure out how large an area we need for the text
    int baseline;
    Size text_size = getTextSize(text, font_face, font_scale, font_thickness, &baseline);
#ifdef DD_CXX11
    cout << "text_size: " << text_size << endl;
#endif
    
    Mat out = bg.clone();
    cv::resize(out,out,Size(text_size.width + 10,text_size.height + 10));
    
    // draw the text
    Point orig_bl(10/2,out.rows - 10/2);
    putText(out,text,orig_bl,font_face,font_scale,font_color,font_thickness);
    
    return out;
  }

  Mat image_text(string text,Vec3b color = Vec3b(255,255,255),Vec3b bg = Vec3b(0,0,0))
  {
    Mat text_area(5,5,DataType<Vec3b>::type,Scalar(bg[0],bg[1],bg[2]));
    
    // return the result
    return image_text(text_area, text, color);
  }
  
  // darker means closer
  Mat eq(Mat&im)
  {
    assert(im.type() == DataType<float>::type);
    // all images should be about 640 by 480 :-)
    im = im.clone();
    
    //printf("+imageeq\n");
    // compute the order statistics
    vector<float> values;
    for(int rIter = 0; rIter <im.rows; rIter++)
      for(int cIter = 0; cIter < im.cols; cIter++)
      {
	float value = im.at<float>(rIter,cIter);
	if(goodNumber(value) /*&& value < params::MAX_Z()*/)
	  values.push_back(value);
      }
    std::sort(values.begin(),values.end());
    auto newEnd = std::unique(values.begin(),values.end());
    values.erase(newEnd,values.end());
    
    // compute an equalized image
    Mat showMe = imageeq_realize(im,values);
    if(values.size() > 5)
    {
      ostringstream quantiles;
      quantiles << " " << values.front();
      for(int iter = 1; iter < 5; ++iter)
      {
	int index = static_cast<double>(iter)/5 * values.size();
	quantiles << " " << values.at(clamp<int>(0,index,values.size()-1));
      }
      quantiles << " " << values.back();
      string txt = quantiles.str();
      Mat txt_im = image_text(txt);
      if(txt_im.cols <= showMe.cols)
	showMe = vertCat(showMe,txt_im);
    }
    else 
      showMe = vertCat(showMe,image_text("BadQuantiles"));
        
    return showMe;
  }
  
  string printfpp(const char* format, ... )
  {
    int size = 128;
    char*char_store = (char*)malloc(size);
    while(true)
    {
      // try to print
      va_list ap;
      va_start(ap,format);
      int n = vsnprintf(char_store,size,format,ap);
      va_end(ap);
      
      // if it worked, return
      if(n > -1 && n < size)
      {
	string result(char_store);
	free(char_store);
	return result;
      }
      
      // otherwise, try again with more space
      if(n > -1)
	size = n+1;
      else
	size *= 2;
      // realloc
      char_store = (char*)realloc(char_store,size);
    }
  }
  
  template<typename D>
  D fromString(const string input)
  {
    std::istringstream iss(input);
    D d;
    iss >> d;
    return d;
  }
  
  Mat read_csv_double(const string&filename,char separator)
  {
    Mat m;
    cout << "++read_csv_double: " << filename << endl;
    // check cache
    {
      ifstream cache(filename + ".bin");
      if(false && cache.good())
      {
	cout << "getting from bin cache" << endl;
	assert(readFileToMat(m, filename + ".bin") == 0);
	return m;
      }
    }
    
    ifstream ifs(filename);
    if(not ifs)
    {
      cout << "error: failed to open " << filename << endl;
      assert(false);
    }

    while(ifs)
    {
      string line;
      std::getline(ifs,line);
      istringstream iss(line);
      vector<double> line_values;
      while(iss)
      {
	double value;
	string svalue;
	getline(iss,svalue,separator);
	value = fromString<double>(svalue);
	line_values.push_back(value);
      } // read all numbers from line
      //cout << "[";
      //for(auto && vec : line_values)
      //cout << "(" << vec << ")";
      //cout << "]" << endl;
      Mat row = Mat(line_values).t();
      if(m.cols == 0 or row.cols == m.cols)
	m.push_back(row);
      else
	break;
    }        

    assert(writeMatToFile(m, filename + ".bin") == 0);
    cout << "--read_csv_double: " << m.size() << endl;
    return m;
  }
  
  static string nyu_base()
  {
    return "/mnt/data/NYU-Hands-v2/";
    //return "/home/jsupanci/workspace/data/NYU-Hands-v2/";
  }
  
  struct NYU_Labels
  {
  public:
    Mat train_us;
    Mat train_vs;
    Mat train_ds;
    Mat test_us;
    Mat test_vs;
    Mat test_ds;
    Mat pred_us;
    Mat pred_vs;
    
    NYU_Labels()
    {
      string testing_dir = nyu_base() + "/test/";
      test_us = read_csv_double(testing_dir + "/joint_1_u.csv");
      test_us = test_us.t();
      //cout << "test_us = " << test_us << endl;
      test_vs = read_csv_double(testing_dir + "/joint_1_v.csv").t();
      test_ds = read_csv_double(testing_dir + "/joint_1_d.csv").t();
      string training_dir = nyu_base() + "/train/";
      train_us = read_csv_double(training_dir + "/train_u.csv").t();
      train_vs = read_csv_double(training_dir + "/train_v.csv").t();
      train_ds = read_csv_double(training_dir + "/train_d.csv").t();
      // NYU predictions
      pred_us = read_csv_double(testing_dir + "/test_predictions_u.csv").t();
      pred_vs = read_csv_double(testing_dir + "/test_predictions_v.csv").t();
    }
  };
  
  static NYU_Labels*nyu_labels()
  {
    static mutex m; lock_guard<mutex> l(m);
    static NYU_Labels *singleton = nullptr;
    if(singleton == nullptr)
      singleton = new NYU_Labels();
    return singleton;
  }

  Mat load_depth(const string&direc, int index)
  {
    Mat coded_depth = imread(printfpp("%s/depth_1_%07d.png",direc.c_str(),index));
    Mat depth(coded_depth.rows,coded_depth.cols,DataType<float>::type,Scalar::all(0));
    float max_depth = -inf;
    for(int rIter = 0; rIter < depth.rows; rIter++)
      for(int cIter = 0; cIter < depth.cols; cIter++)
      {
	// Note: In each depth png file the top 8 bits of depth are packed into the green channel and the lower 8 bits into blue.
	Vec3b coded_pix = coded_depth.at<Vec3b>(rIter,cIter);
	float decoded_depth = 256*coded_pix[1] + coded_pix[0]; // take the green and blue
	float zf = static_cast<float>(decoded_depth)/10;	
	//if(zf > 200)
	//zf = inf;
	depth.at<float>(rIter,cIter) = zf;
	max_depth = std::max<float>(max_depth,zf);
      }

    return depth;
  }

  Mat imroi(const Mat&im,Rect roi)
  {
    int pad_top    = std::max(-roi.y,0);
    int pad_left   = std::max(-roi.x,0);
    int pad_bottom = std::max(roi.br().y - im.rows,0);
    int pad_right  = std::max(roi.br().x - im.cols,0);

    Mat result; cv::copyMakeBorder(im,result,
				   pad_top,pad_bottom,pad_left,pad_right,
				   cv::BORDER_REPLICATE);

    return result(Rect(Point(roi.x + pad_left,roi.y + pad_top),roi.size())).clone();
  }

  Point_<double> point_affine(Point_<double> point, const Mat& affine_transform)
  {
    Mat at; affine_transform.convertTo(at,DataType<float>::type);
    assert(at.type() == DataType<float>::type);
    double m11 = at.at<float>(0,0);
    double m12 = at.at<float>(0,1);
    double m13 = at.at<float>(0,2);
    double m21 = at.at<float>(1,0);
    double m22 = at.at<float>(1,1);
    double m23 = at.at<float>(1,2);
    return Point_<double>(
      m11*point.x+m12*point.y+m13,
      m21*point.x+m22*point.y+m23);
  }

  Vec3d vec_affine(Vec3d v, const Mat&affine)
  {
    Point2d pt = point_affine(Point2d(v[0],v[1]),affine);
    return Vec3d(pt.x,pt.y,v[2]);
  }
  
  cv::Mat imscale(const cv::Mat&im, double sf, Mat&atMat)
  {
    Point2f center = Point2f(im.cols/2,im.rows/2);
    atMat = cv::getRotationMatrix2D(center,0,sf);
    float z_max = extrema(im).max;
    Mat scaled; cv::warpAffine(im,scaled,atMat,im.size(),
			       jvl::DEPTH_INTER_STRATEGY,cv::BORDER_CONSTANT,Scalar::all(z_max));
    return scaled;
  }
  
  Mat imrotate(const Mat&m, double angle_in_radians, Mat&rotMat, Point2f center)
  {
    // handle default value
    if(center == Point2f())
      center = Point2f(m.cols/2,m.rows/2);

    rotMat = cv::getRotationMatrix2D(center,rad2deg(angle_in_radians),1);    
    Mat rotated;    
    if(m.type() == DataType<float>::type)
    {
      //cv::warpAffine(m,rotated,rotMat,m.size(),params::DEPTH_INTER_STRATEGY,cv::BORDER_REPLICATE);
      float z_max = extrema(m).max;
      cv::warpAffine(m,rotated,rotMat,m.size(),jvl::DEPTH_INTER_STRATEGY,cv::BORDER_CONSTANT,Scalar::all(z_max));
    }
    else
      cv::warpAffine(m,rotated,rotMat,m.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    return rotated;
  }
  
  Rect_<double> rectFromCenter(Point2d center, Size_<double> size)
  {
    Point2d tl(center.x-size.width/2.0,
	       center.y-size.height/2.0);
    return Rect_<double>(tl,size);
  }
  
  Rect rectResize(Rect r, float xScale, float yScale)
  {
    Point center(r.x+r.width/2,r.y+r.height/2);
    Size size(r.width*xScale,r.height*yScale);
    return rectFromCenter(center,size);
  }

  //constexpr float fx = 224.502; // ICL/PXC
  constexpr float fx =  525; // NYU/Kinect
  constexpr float metric_size_factor = fx*metric_size;

  Rect bounding_box(Vec3d center)
  {
    return rectFromCenter(Point2d(center[0],center[1]),
			  Size(metric_size_factor/center[2],
			       metric_size_factor/center[2]));    
  }
  
  RegExample load_nyu_datum(int idx,const string&dir,const Mat&us,const Mat&vs,const Mat&ds,bool predictions = false)
  {
    // Stage #1 : get the data    
    Mat Z = load_depth(dir,idx);
    //cv::imwrite(printfpp("out/Zraw%d.png",idx),eq(Z));

    vector<Vec3d> uvds;
    Rect handBB;
    for(int kpIter = 0; kpIter < us.rows; ++kpIter)
    {
      double u = us.at<double>(kpIter,idx);
      double v = vs.at<double>(kpIter,idx);
      double d = ds.at<double>(kpIter,idx)/10;
      // Point2d pt(u,v);
      // if(handBB == Rect())
      // 	handBB = Rect(pt,Size(1,1));
      // else
      // 	handBB |= Rect(pt,Size(1,1));
      uvds.push_back(Vec3d(u,v,d));
    }
    //handBB = rectResize(handBB,1.1,1.1);
    int id_middle_finger_base = 17;
    Point2d center(uvds.at(id_middle_finger_base)[0],uvds.at(id_middle_finger_base)[1]);
    vector<float> all_depths;
    for(auto && uvd : uvds)
      all_depths.push_back(uvd[2]);
    std::sort(all_depths.begin(),all_depths.end());
    float center_depth = all_depths.at(all_depths.size()/2);//uvds.at(id_middle_finger_base)[2];
    //cout << "d = " << d << endl;
    handBB = bounding_box(Vec3d(center.x,center.y,center_depth));

    // Stage #2: normalize the UV coordinate frame
    //cout << "handBB = " << handBB << endl;
    for(int kpIter = 0; kpIter < us.rows; ++kpIter)
    {
      uvds.at(kpIter)[0] -= handBB.x;
      uvds.at(kpIter)[1] -= handBB.y;
      uvds.at(kpIter)[2] -= center_depth;
    }
    
    // Stage #3: normalize the z coordinate frame
    Z = imroi(Z,handBB);
    //cv::imwrite(printfpp("out/Zroi%d.png",idx),eq(Z));
    // compute statistics about the image patch
    // now clamp the extracted volume
    assert(goodNumber(center_depth));
    vector<bool> kp_used(uvds.size(),false);
    for(int yIter = 0; yIter < Z.rows; ++yIter)
      for(int xIter = 0; xIter < Z.cols; ++xIter)
      {
	float&z = Z.at<float>(yIter,xIter);	
	if(!goodNumber(z) && !predictions)
	{
	  cout << "(INFO) bad number" << endl;
	  return RegExample{};
	}
	z = clamp<float>(-15,
			 z - center_depth,
			 +15)/15;

	// find the nearest keypoint
	auto dist = [center_depth](const Vec3d&x1, const Vec3d&x2)
	  {
	    Vec3d diff = x1 - x2;
	    diff[0] *= center_depth/fx;
	    diff[1] *= center_depth/fx;
	    return std::sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
	  };
	float min_dist_2_a_keypoit = inf;
	size_t closest_kp;
	for(size_t kp_iter = 0; kp_iter < uvds.size(); ++kp_iter)
	{
	  auto && uvd = uvds.at(kp_iter);
	  double d = dist(uvd,Vec3d(xIter,yIter,z));
	  if(d < min_dist_2_a_keypoit)
	  {
	    min_dist_2_a_keypoit = d;
	    closest_kp = kp_iter;
	  }
	}
	if(min_dist_2_a_keypoit > 5)
	  z = +1;
	else
	  kp_used[closest_kp] = true;
	
	assert(goodNumber(z));
      }
    int used_kps = 0;
    for(auto && f_kp_used : kp_used)
      used_kps += f_kp_used;
    if(used_kps < uvds.size() / 2  && !predictions)
    {
      //cout << "(INFO) Not enough keypoints used" << endl;
      return RegExample{};
    }

    // draw the loaded example for training
    if(false)
    {
      Mat vis = eq(Z);
      cout << "Keypoint # = " << us.rows << endl;
      for(int kpIter = 0; kpIter < us.rows; ++kpIter)
      {
	Point center(uvds.at(kpIter)[0],uvds.at(kpIter)[1]);
	//cv::circle(vis,,5,Scalar(0,255,0));
	cv::putText(vis,printfpp("%d",kpIter),center, FONT_HERSHEY_SIMPLEX,.5,Scalar(0,255,0));
      }    
      cv::imwrite(printfpp("out/Z%d.png",idx),vis);
    }
    
    return RegExample{handBB,Z,uvds,idx};
  }

  int set_size(const std::string&set_id)
  {
    if(set_id == "training")
      return nyu_labels()->train_us.cols;
    else if(set_id == "testing" || set_id == "predicting")
      return nyu_labels()->test_us.cols;
    else
      throw std::runtime_error("set_size failed");
  }
  
  RegExample load_nyu_datum(const string&set_id, int index)
  {
    RegExample ex;
    if(set_id == "training")
    {
      int idx = (index == -1)?rand() % nyu_labels()->train_us.cols:index;
      string training_dir = nyu_base() + "/train/";
      const Mat&us = nyu_labels()->train_us;
      const Mat&vs = nyu_labels()->train_vs;
      const Mat&ds = nyu_labels()->train_ds;
      ex = load_nyu_datum(idx,training_dir,us,vs,ds);
    }
    else if(set_id == "testing" || set_id == "predicting")
    {
      int idx = (index == -1)?rand() % nyu_labels()->test_us.cols:index;
      string test_dir = nyu_base() + "/test/";
      const Mat&us = nyu_labels()->test_us;
      const Mat&vs = nyu_labels()->test_vs;
      const Mat&ds = nyu_labels()->test_ds;
      ex = load_nyu_datum(idx,test_dir,us,vs,ds,set_id == "predicting");
      if(set_id == "predicting")
      {
	for(int iter = 0; iter < ex.uvd.size(); ++iter)
	{
	  ex.uvd.at(iter)[0] = nyu_labels()->pred_us.at<double>(iter,idx) - ex.handBB.tl().x;
	  ex.uvd.at(iter)[1] = nyu_labels()->pred_vs.at<double>(iter,idx) - ex.handBB.tl().y;
	}
      }
    }
    else
      assert(false);

    if(ex.Z.empty())
      return load_nyu_datum(set_id,index);
    else
      return ex;
  }

  Mat tileCat(vector< Mat > ms, bool number)
  {
    // compute the geometry
    int n_rows = ceil(std::sqrt((double)ms.size()));
    int n_cols = ceil((double)ms.size()/n_rows);
    
    // build each row independently
    vector<Mat> rows(n_rows);
    for(int iter = 0; iter < ms.size(); iter++)
    {
      // debug
      Mat tile_here = number?vertCat(ms[iter],image_text(printfpp("%d",iter))):ms[iter];
      int row = iter % n_cols;
      rows[row] = horizCat(rows[row],tile_here);
    }
    
    // combine the rows to get the entire image
    Mat tiles;
    for(int row = 0; row < rows.size(); row++)
      tiles = vertCat(tiles,rows[row]);
    
    return tiles;
  }

  void breakpoint()
  {
    asm volatile ("int3;");
  }

  Extrema extrema(const Mat& im)
  {
    Extrema result;
    cv::minMaxLoc(im,
		  &result.min,
		  &result.max,
		  &result.minLoc,
		  &result.maxLoc);
    return result;
  }

  double rad2deg(double rad)
  {
    return rad*180.0/M_PI;
  }
  
  double deg2rad(double deg)
  {
    return deg*M_PI/180.0;
  }

  float sample_in_range(float min, float max)
  {
    float U = ((float)rand())/RAND_MAX;
    float V = (max-min)*U+min;
    //std::cout << "sample = " << V << std::endl;
    return V;
  }

  ///
  /// SECTION: ICL Metadata
  ///
  static string icl_base()
  {
    const string dir_base = "/home/jsupanci/workspace/data/ICL_HANDS2/";
    return dir_base;
  }
  
  struct ICL_Dataset
  {
  protected:
    vector<string> lines;
    string subset;
    
  public:
    size_t instances() const
    {
      assert(lines.size() > 0);
      return lines.size();
    }

    RegExample load(size_t index)
    {
      string annotation = lines.at(index);
      RegExample ex;
      ex.index = index;
      
      // parse the annotation
      vector<Vec3d> labels;
      istringstream iss(annotation);
      string filename; iss >> filename; // discard.
      //cout << "annotation " << annotation << endl;
      //cout << "filename " << filename << endl;
      while(iss)
      {
	double u; iss >> u;
	double v; iss >> v;
	double d; iss >> d; d /= 10;
	labels.push_back(Vec3d(u,v,d));
      }
      ex.uvd.resize(36);
      for(auto active_keypoint : active_keypoints)
      {	
	ex.uvd.at(active_keypoint) = labels.at(icl_correspondence.at(active_keypoint));
      }

      // compute the handbb
      Vec3d center = labels.at(0);
      ex.handBB = bounding_box(Vec3d(center[0],center[1],center[2]));      
      //cout << "handbb = " << ex.handBB << endl;
      for(int iter = 0; iter < ex.uvd.size(); ++iter)
      {
	if(ex.uvd.at(iter) != Vec3d())
	{
	  ex.uvd.at(iter)[0] -= ex.handBB.x;
	  ex.uvd.at(iter)[1] -= ex.handBB.y;
	  //kp[0] -= center[0];
	  //kp[1] -= center[1];
	  ex.uvd.at(iter)[2] -= labels.at(0)[2];	  
	}
      }
      
      // load the dpeth map
      string base_path = icl_base() + subset + "/Depth/";
      string frame_file = base_path + filename;
      Mat depth = imread(frame_file,-1);
      if(depth.empty())
	throw std::runtime_error("failed to load depth: " + frame_file);
      depth.convertTo(depth,DataType<float>::type);
      depth /= 10;
      for(int rIter = 0; rIter < depth.rows; ++rIter)
	for(int cIter = 0; cIter < depth.cols; ++cIter)
	{
	  float & d = depth.at<float>(rIter,cIter);
	  if(!goodNumber(d))
	    d = 1;
	  else
	    d = clamp<float>(-15,
			     d - center[2],
			     +15)/15;
	}
      ex.Z = imroi(depth,ex.handBB).clone();
      ex.src = frame_file;
      
      return ex;
    }
    
    ICL_Dataset(const string&subset) : subset(subset)
    {
      ifstream ifs("/home/jsupanci/workspace/data/ICL_HANDS2/" + subset + "/labels.txt");
      while(ifs)
      {
	string line; std::getline(ifs,line);
	istringstream iss(line);
	string file; iss >> file;
	string frame_file = icl_base() + "/" + subset + "/Depth/" + file;
	boost::algorithm::trim(file);
      
	bool file_exists = boost::filesystem::exists(frame_file) and file != "";
	if(file_exists)
	{
	  //Mat depth = imread(frame_file,-1);
	  //if(depth.type() == DataType<uint16_t>::type)
	  lines.push_back(line);
	}
      }

      cout << "ICL_Dataset: " << subset << " loaded " << lines.size() << endl;
      if(lines.size() <= 0)
	throw std::runtime_error("Failed to load");
    };
  }; //icl_training_dataset("Training"), icl_testing_dataset("Testing");

  static ICL_Dataset*icl_training_dataset()
  {
    return nullptr;
  }
  
  static ICL_Dataset*icl_testing_dataset()
  {
    return nullptr;
  }
  
  RegExample load_icl_datum(const std::string&set_id,int index)
  {
    // select dataset
    ICL_Dataset*dataset = nullptr;
    if(set_id == "training")
      dataset = icl_training_dataset();
    else if(set_id == "testing")
      dataset = icl_testing_dataset();
    else
      throw std::runtime_error("invalid set identifier");

    // load instances
    //cout << "dataset->instances(): " << dataset->instances() << endl;
    int idx = (index == -1)?rand() % dataset->instances():index;
    return dataset->load(idx);
  }

  ///
  /// Load the regex from a directory at random
  ///
  std::vector< std::string > regex_matches(std::string str, boost::regex re)
  {
    vector<string> matches;
    
    boost::sregex_token_iterator iter(str.begin(),str.end(),re,0);
    boost::sregex_token_iterator end;
    
    for(; iter != end; ++iter)
    {
      matches.push_back(*iter);
    }
    
    return matches;
  }

  
  RegExample load_direc_datum(const string&direc)
  {
    // choose an option
    vector<string> options = find_files(direc,boost::regex(".*\\.exr"));
    std::sort(options.begin(),options.end());
    int index = rand()%options.size();
    RegExample ex;
    ex.index = index;
    ex.src = options.at(index);

    //
    ex.Z = cv::imread(ex.src,-1);
    ex.Z.convertTo(ex.Z,DataType<float>::type);
    ex.handBB = Rect(Point(0,0),ex.Z.size());
    
    // read uvd
    string id = regex_matches(ex.src,boost::regex("\\d+")).back();
    ifstream ifs(direc + "/labels_" + id + + ".txt");
    if(!ifs)
      throw std::runtime_error("couldn't open labels");
    ex.uvd.resize(36);
    vector<Vec3d> in_uvd;
    for(int iter = 0; iter < 5; ++iter)
    {
      double u; ifs >> u;
      double v; ifs >> v;
      in_uvd.push_back(Vec3d(u,v,0));
    }
    for(int iter = 0; iter < 6; ++iter)
      ex.uvd.at(iter) = in_uvd.at(0);
    for(int iter = 6; iter < 12; ++iter)
      ex.uvd.at(iter) = in_uvd.at(1);
    for(int iter = 12; iter < 18; ++iter)
      ex.uvd.at(iter) = in_uvd.at(2);
    for(int iter = 18; iter < 24; ++iter)
      ex.uvd.at(iter) = in_uvd.at(3);
    for(int iter = 24; iter < 36; ++iter)
      ex.uvd.at(iter) = in_uvd.at(4);

    return ex;
  }

  ///
  /// util_file
  ///
  vector< string > find_files(string dir, boost::regex regex,bool recursive)
  {
    printf("allStems %s\n",dir.c_str());
    vector<string> matches;
    boost::filesystem::path direc(dir);
    int file_count = 0;
    for(boost::filesystem::directory_iterator fIter(direc);
	fIter != boost::filesystem::directory_iterator();
	fIter++)    
    {
      file_count++;
      string file = fIter->path().string();	
      if(boost::regex_match(fIter->path().string(),regex))
      //if(fIter->path().extension().string() == ext)
      {	
	matches.push_back(file);
      }
      if(boost::filesystem::is_directory(file) and recursive)
      {
	vector<string> sub_matches = find_files(file,regex,recursive);
	matches.insert(matches.end(),sub_matches.begin(),sub_matches.end());
      }
    }
    std::sort(matches.begin(),matches.end());
    printf("allStems %d of %d\n",(int)matches.size(),file_count);
    return matches;
  }  
}
