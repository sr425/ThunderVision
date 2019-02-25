#pragma once

#include "Tensor.h"
#include "FilterUtil.h"

namespace ThunderVision
{
class GaussianBlur
{
  public:
	template <typename T>
	Tensor<T> ApplyGaussian(const Tensor<T> &tensor, const double sigma)
	{
		int size = static_cast<int>(2.0 * sigma);
		int filterSize = 2 * size;
		if (filterSize % 2 == 0)
			filterSize++;
		return ApplyGaussian(tensor, sigma, filterSize);
	}

	template <typename T>
	Tensor<T> ApplyGaussian(const Tensor<T> &tensor, const double sigma, const int filterSize)
	{
		if (filterSize % 2 != 1)
		{
			throw new ThunderException("Invalid kernel size, has to be uneven");
		}
		auto mask = generateGaussianMask1D<double>(sigma, filterSize);

		auto filteredX = filterUtility.ApplyFilter<1, 0, double>(tensor, mask);
		return filterUtility.ApplyFilter<0, 1, T>(filteredX, mask);
	}

  private:
	FilterUtil filterUtility;

	template <typename T>
	Tensor<T> generateGaussianMask1D(const double sigma, const size_t filterSize)
	{
		if (filterSize % 2 != 1)
		{
			throw new ThunderException("The filtersize has to be uneval");
		}

		Tensor<T> filter({filterSize});
		int64_t filterSizeHalf = (filterSize - 1) >> 1;

		//double preFactor = 1.0 / (sigma * sqrt(2 * M_PI));

		T sumOfValues = 0.0;
		for (int x = 0; x <= filterSizeHalf; ++x)
		{
			filter[x + filterSizeHalf] = static_cast<T>(exp(-1 * (x * x) / (2 * sigma * sigma)));
			sumOfValues += filter[x + filterSizeHalf];
		}

		for (int x = 0; x < filterSizeHalf; ++x)
		{
			filter[x] = filter[filterSize - 1 - x];
			sumOfValues += filter[x];
		}

		T factor = 1.0 / sumOfValues;

		for (int i = 0; i < filterSize; ++i)
		{
			filter[i] = factor * filter[i];
		}
		return filter;
	}
};
} // namespace ThunderVision