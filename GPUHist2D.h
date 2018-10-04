#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "GPUVecArray.h"

template <class T, int xDim, int yDim, int max_depth> struct histogram2D{

  __host__ __device__
  void init(float xMin, float xMax, float yMin, float yMax)
  {
    limits_[0]=xMin;
    limits_[1]=xMax;
    limits_[2]=yMin;
    limits_[3]=yMax;
    for(int i =0; i < xDim*yDim; i++)
      data_[i].reset();
  }

  __host__  __device__
  bool fillBin(float x, float y, const T idx)
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
    data_[xBin + yBin*xDim].push_back(idx);
    if(data_[xBin + yBin*xDim].push_back(idx) != -1)
      return true;
    else
      return false;
  }

  __host__ __device__
  int computeXBinIndex(float x)
  {
    int xIndex = floor((std::abs(x) - limits_[0]) / binSize_);
    return xIndex;
  }

  __host__ __device__
  int computeYBinIndex(float y) 
  {
    int yIndex = floor((y + limits_[3]) / binSize_);
    return yIndex;
  } 

  __host__ __device__
  void printBinContent(float x, float y)
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
    
    for(int i = 0; i < max_depth; i++)
      printf("* Element: %f %f %d %d\n", x, y, xBin + yBin*xDim, data_[xBin + yBin*xDim][i]);
  }

  __host__ __device__
  void printBinContent2(int x, int y)
  {
    for(int i = 0; i < max_depth; i++)
      printf("*Element", data_[x + y*xDim][i]);
   }

  __host__ __device__
  int getBinId(float x, float y)
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
    //std::cout << "xBin: " << xBin << " yBin: " << yBin << std::endl;
    //std::cout << "Abs(eta): " << std::abs(x) << std::endl;
    return xBin + yBin*xDim;
  }

  __host__ __device__
  GPU::VecArray<T, max_depth> getBinContent(float x, float y)
  {
    int xBin = computeXBinIndex(x);
    int yBin = computeYBinIndex(y);
    return data_[xBin + yBin*xDim];
  }

  __host__ __device__
  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax)
  {
    int xBinMin = xMin <= limits_[0] ? computeXBinIndex(limits_[0]) : computeXBinIndex(xMin);
    int xBinMax = xMax >= limits_[1] ? computeXBinIndex(limits_[1]) : computeXBinIndex(xMax);
    int yBinMin = yMin <= limits_[2] ? computeYBinIndex(limits_[2]) : computeYBinIndex(yMin);
    int yBinMax = yMax >= limits_[3] ? computeYBinIndex(limits_[3]) : computeYBinIndex(yMax);
    std::array<int, 4> box;
    box[0] = xBinMin;
    box[1] = xBinMax - xBinMin + 1;
    box[2] = yBinMin;
    box[3] = yBinMax - yBinMin+ 1;
    return box;
  }
  
  inline constexpr int size() const { return data_.size(); }
  inline constexpr GPU::VecArray<T, max_depth>& operator[](int i) { return data_[i]; }
  
  GPU::VecArray<GPU::VecArray<T, max_depth>, xDim*yDim> data_;
  std::array<float, 4> limits_;
  float binSize_ = 0.05;  
};
