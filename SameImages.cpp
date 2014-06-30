/*
 *	Author: John Hany
 *	Contact: johnhany@163.com
 *	Website: http://johnhany.net
 *	Source code updates: https://github.com/johnhany/SameImages
 *	Using OpenCV and dirent.h
 *	Under no license currently. So, do whatever you want with it:)
 */

#include <Windows.h>
#include <tchar.h>
#include <stdio.h>
#include <strsafe.h>
#include <iostream>
#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dirent.h"

using namespace cv;
using namespace std;

#define VALID_TYPE_COUNT 3
#define STAMP_WIDTH		8
#define STAMP_HEIGHT	8
#define STAMP_SIZE		STAMP_WIDTH*STAMP_HEIGHT
#define MAX_CLASS_NUM	10

enum DeleteType {DELETE_FIRST = 1, DELETE_LAST, DELETE_LARGE, DELETE_SMALL};
enum CompareType {EXACT_SAME = 1, MOST_MATCHES, NEAREST_DISTANCE};

const char hex_char[] = {
	'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'
};
const int hex_int[] = {
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

void readPicture(string, vector<Mat> &);
void generateStamps(vector<Mat> &, vector<string> &, vector<pair<int,float> > &);
void roughClassify(vector<pair<int,float> > &, vector<string> &, vector<vector<pair<string,int> >> &, int);
void comparePictures(vector<vector<pair<string,int> >> &, vector<pair<int,int> > &, CompareType);
void deleteSameFile(vector<string> &, vector<string> &, vector<pair<int,int> > &, DeleteType);
int hexStrCompare(string, string);
void renamePictures(vector<string> &, string, vector<string> &, vector<string> &);

int main(int argc, char **argv)
{
	string root_path = "H:\\dataSets\\sample_image";
	const char * root_path_c = root_path.c_str();

	vector<string> file_names;
	vector<string> file_paths;
	int file_count = 0;

	//Do not add ".jpeg" to this list, since its length is 5, rather than 4.
	//For better performance, the first image format(".jpg") should be the most common one in your directory.
	vector<string> valid_type(VALID_TYPE_COUNT);
	valid_type.push_back(".jpg");
	valid_type.push_back(".png");
	valid_type.push_back(".bmp");

	vector<Mat> stamp_img;
	vector<string> stamp_str;
	vector<pair<int,float> > stamp_stats;
	vector<vector<pair<string,int> >> stamp_class;
	vector<pair<int,int> > out_same;

	DIR *dir;
	struct dirent *ent;
	if((dir = opendir(root_path_c)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			string cur_file(ent->d_name);
			for(vector<string>::iterator t=valid_type.begin(); t!=valid_type.end(); t++) {
				if(cur_file.size() > 4 && cur_file.compare(cur_file.size()-4, 4, *t) == 0) {
					file_count ++;
					file_names.push_back(cur_file);
					file_paths.push_back(root_path+"\\"+cur_file);

					readPicture(file_paths.back(), stamp_img);

					break;
				}
			}
		}
		closedir (dir);
	}else {
		cout << "Could not open directory" << endl;
		return EXIT_FAILURE;
	}

	stamp_str.reserve(file_count);
	stamp_stats.reserve(file_count);

	cout << "Number of pictures: " << file_count << endl;

	generateStamps(stamp_img, stamp_str, stamp_stats);

	roughClassify(stamp_stats, stamp_str, stamp_class, file_count);
	
	comparePictures(stamp_class, out_same, NEAREST_DISTANCE);

	deleteSameFile(file_names, file_paths, out_same, DELETE_SMALL);

	renamePictures(file_paths, root_path+"\\", stamp_str, valid_type);

	cout << "ALL DONE." << endl;
	cin.get();

	return 0;
}

//Read image and normalize it to 8*8, with gray value from 0 to 7
void readPicture(string path, vector<Mat> &stamps)
{
	Mat srcImg = imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	Mat norImg(STAMP_WIDTH, STAMP_HEIGHT, CV_8U);
	Mat tmpImg(64, 64, CV_8U);
	if (!srcImg.data) {
		cout << "The image " << path << " could not be loaded." << endl;
		waitKey(0);
		return;
	}

	resize(srcImg, norImg, Size(STAMP_WIDTH,STAMP_HEIGHT), 0, 0, INTER_LINEAR);

	Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.data; 
	for(int i=0; i<256; i++)
		p[i] = i/32;
	LUT(norImg, lookUpTable, norImg);

	stamps.push_back(norImg);
}

//Transform 32 8-bit values to 16 16-bit value string
void generateStamps(vector<Mat> & img, vector<string> & str, vector<pair<int,float> > & stats)
{
	for(vector<Mat>::iterator p=img.begin(); p!=img.end(); p++) {
		stringstream stream;
		int width = (*p).cols;
		int height = (*p).rows;
		int gray, mean = 0;
		float dev = 0;
		int tmp[STAMP_SIZE] = {0};

		uchar *psrc;
		for(int j=0, idx=0; j<height; j++) {
			psrc = (*p).ptr<uchar>(j);
			for(int i=0; i<width; i+=2, idx++) {
				gray = psrc[i] + psrc[i+1];
				mean += gray;
				tmp[idx] = gray;
				stream << std::hex << gray;
			}
		}
		mean = cvRound((double)mean/height/width);
		for(int i=0; i<STAMP_SIZE; i++) {
			dev += (tmp[i] - mean) * (tmp[i] - mean);
		}
		dev /= height*width;
		stats.push_back(make_pair<int,float>(mean, dev));
		str.push_back(stream.str());
	}
}

//For better performance, classify all the images with K-means
void roughClassify(vector<pair<int,float> > & stats, vector<string> & stamp, vector<vector<pair<string,int> >> & result, int amount)
{
	int num_class = amount / 10;
	if(num_class > MAX_CLASS_NUM)
		num_class = MAX_CLASS_NUM;
	int dev_class[MAX_CLASS_NUM] = {0};

	cout << "Number of classes:" << num_class << endl;

	Mat src_data(amount, 1, CV_32F);
	Mat out_labels, out_centers;
	int j = 0;
	for(vector<pair<int,float> >::iterator i=stats.begin(); i!=stats.end(); i++, j++) {
		src_data.at<float>(j) = (*i).second;
	}

	kmeans(src_data, num_class, out_labels, TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.5), 3, KMEANS_RANDOM_CENTERS, out_centers);

	for(int c=0; c<num_class; c++) {
		vector<pair<string,int> > tmp_vec;
		for(int i=0; i<amount; i++) {
			if(out_labels.at<int>(i) == c) {
				tmp_vec.push_back(make_pair(stamp[i],i));
			}
		}
		result.push_back(tmp_vec);
	}
}

//Find same images within each class
void comparePictures(vector<vector<pair<string,int> >> & src_vec, vector<pair<int,int> > & out_same, CompareType flag)
{
	int same_count = 0;
	int num_class = src_vec.size();
	vector<vector<pair<string,int> >>::iterator idx_class = src_vec.begin();

	for(int c=0; c<num_class; c++) {
		int num_str = (*idx_class).size();
		vector<pair<string,int> >::iterator idx_str1 = (*idx_class).begin();
		vector<pair<string,int> >::iterator idx_str2 = (*idx_class).begin();
		for(int i=0; i<num_str; i++) {
			idx_str1 = (*idx_class).begin() + i;
			for(int j=i+1; j<num_str; j++) {
				idx_str2 = (*idx_class).begin() + j;

				if(flag == EXACT_SAME) {
					//If you need two strings to be exactly the same
					if(((*idx_str1).first).compare((*idx_str2).first) == 0) {
						out_same.push_back(make_pair((*idx_str1).second, (*idx_str2).second));
						same_count ++;
					}
				}else if(flag == MOST_MATCHES) {
					//If you need most of characters of two strings to be the same
					//0.8 * 32 = 25
					if(hexStrCompare((*idx_str1).first, (*idx_str2).first) >= 25) {
						out_same.push_back(make_pair((*idx_str1).second, (*idx_str2).second));
						same_count ++;
					}
				}else if(flag == NEAREST_DISTANCE) {
					//Characters with similar hex values will be considered to be the same
					map<char, int> hex_map;
					for(int i=0; i<16; i++) {
						hex_map.insert(pair<char,int>(hex_char[i], hex_int[i]));
					}

					const char * cstr1 = (*idx_str1).first.c_str();
					const char * cstr2 = (*idx_str2).first.c_str();
					int len = (*idx_str1).first.length();
					int vote = 0;
					for(int i=0; i<len; i++) {
						if(cstr1[i] == cstr2[i]) {
							vote += 2;
						}else if(abs(hex_map.find(cstr1[i])->second - hex_map.find(cstr2[i])->second) == 1) {
							vote += 1;
						}
					}

					//0.8 * 64 = 51
					if(vote > 51) {
						out_same.push_back(make_pair((*idx_str1).second, (*idx_str2).second));
						same_count ++;
					}
				}else {
					cout << "comparePictures(): argument \'flag\' is not valid" << endl;
					cin.get();
					return;
				}
			}
		}

		idx_class++;
	}
	
	cout << "Same pictures found: " << same_count << endl;
}

//Calculate how many characters are same in two strings
int hexStrCompare(string str1, string str2)
{
	int len = str1.length();
	if(len != str2.length()) {
		cout << "Cannot minus two hex strings" << endl;
		cin.get();
		return -1;
	}

	string result;
	const char *cstr1 = str1.c_str();
	const char *cstr2 = str2.c_str();

	int vote = 0;
	for(int i=0; i<len; i++) {
		if(cstr1[i] == cstr2[i]) {
			vote ++;
		}
	}

	return vote;
}

//Delete the redundant images
void deleteSameFile(vector<string> & files, vector<string> & paths, vector<pair<int,int> > & same_pairs, DeleteType flag)
{
	cout << "Deleting..." << endl;

	vector<string>::iterator file_name = files.begin();

	for(vector<pair<int,int> >::iterator i=same_pairs.begin(); i!=same_pairs.end(); i++) {
		vector<string>::iterator idx = paths.begin();
		int shift = 0;

		if(flag == DELETE_FIRST) {
			shift = (*i).first;
		}else if(flag == DELETE_LAST) {
			shift = (*i).second;
		}else {
			Mat img1 = imread((*(idx+(*i).first)).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			Mat img2 = imread((*(idx+(*i).second)).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
			int size1 = img1.rows * img1.cols;
			int size2 = img2.rows * img2.cols;

			if(flag == DELETE_LARGE) {
				if(size1 > size2)
					shift = (*i).first;
				else
					shift = (*i).second;
			}else if(flag == DELETE_SMALL){
				if(size1 > size2)
					shift = (*i).second;
				else
					shift = (*i).first;
			}else {
				cout << "deleteSameFile(): argument \'flag\' is not valid" << endl;
				cin.get();
				return;
			}
		}
		idx += shift;
/*
		//Be careful with this!
		if(remove((*idx).c_str()) != 0) {
			cout << "Cannot delete file: " << (*idx) << endl;
			cin.get();
		}
*/
	}
}

//Rename all the rest of the images with its stamp string
void renamePictures(vector<string> & paths, string root_path, vector<string> & stamps, vector<string> & types)
{
	cout << "Renaming..." << endl;

	int amount = paths.size();

	for(int id=0; id<amount; id++) {
		vector<string>::iterator idx_path = paths.begin();
		vector<string>::iterator idx_name = stamps.begin();

		Mat testImg = imread((*(idx_path+id)).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		if(testImg.data) {
			string new_name = root_path;
			string file_name = (*(idx_path+id)).substr(root_path.length(), string::npos);

			vector<string>::iterator cur_type = types.begin();
			for(int t=0; t<types.size(); t++) {
				if(file_name.compare(file_name.size()-4, 4, *(cur_type+t)) == 0) {
					new_name += *(idx_name+id) + *(cur_type+t);
				}
			}
/*
			//Be careful with this!
			rename((*(idx_path+id)).c_str(), new_name.c_str());
*/
		}
	}
}
