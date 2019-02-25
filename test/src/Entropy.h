#pragma once
#include <Tensor.h>
#include <cstdint>
#include <cmath>
#include "ImageLoader.h"

using namespace ThunderVision;

class Entropy
{
  public:
	template <typename T>
	Tensor<float> ComputeEntropy(const Tensor<T> &image, const size_t patch_w, const size_t patch_h)
	{
		if (!(patch_h % 2 != 0 && patch_w % 2 != 0))
		{
			throw new ThunderException("Patch size has to be uneven");
		}

		if (image.GetRank() != 3 && image.GetRank() != 2)
			throw new ThunderException("At the moment only images of rank 2 or 3 are allowed");

		const size_t width = static_cast<size_t>(image.GetDimension(1));
		const size_t height = static_cast<size_t>(image.GetDimension(0));
		const size_t channels = image.GetRank() == 2 ? 1 : image.GetDimension(2);

		Tensor<int> imageScaled = quanticeImageValues(image, 255);

		Tensor<float> entropies({height, width, channels });

		auto patch_h_h = static_cast<int64_t>((patch_h - 1) / 2);
		auto patch_w_h = static_cast<int64_t>((patch_w - 1) / 2);

		for (size_t c = 0; c < channels; c++)
		{
		for (size_t x = 0; x < width; x++)
		{
			for (size_t y = 0; y < height; y++)
			{
				entropies[(y * width + x)*channels + c]=computeEntropyPos(imageScaled, x, y, c, patch_w_h, patch_h_h);
							}
		}
		}
		return entropies;
	}

  private:
	int64_t imageValueScale = 255;

	inline float computeEntropyPos(const Tensor<int> &image, const size_t x, const size_t y, const size_t c, const int64_t patch_w_h, const int64_t patch_h_h)
	{
		const size_t width = static_cast<size_t>(image.GetDimension(1));
		const size_t height = static_cast<size_t>(image.GetDimension(0));
		const size_t channels = image.GetRank() == 2 ? 1 : image.GetDimension(2);

		size_t N = 0;
		int maxValue = ceil(imageValueScale);
		int *counter = new int[maxValue+1];
		for (size_t i = 0; i <= maxValue; i++)
		{
			counter[i] = 0;
		}

		//for (int64_t c_t = 0; c_t < channels; c_t++)
		//{
			for (int64_t x_t = -patch_w_h; x_t <= patch_w_h; x_t++)
			{
				for (int64_t y_t = -patch_h_h; y_t <= patch_h_h; y_t++)
				{
					int x_pos = x_t + x;
					int y_pos = y_t + y;
					if (x_pos < 0 || x_pos >= width || y_pos < 0 || y_pos >= height)
					{
						continue;
					}

					auto pixValue = image[y_pos *width* channels + x_pos * channels + c];
					counter[pixValue]++;
					N++;
				}
			}
		//}


		//Compute entropy by using occurence counts of single values
		double probabilitySum = 0.0;
		double N_d = static_cast<double>(N);
		for (size_t i = 0; i <= maxValue; i++)
		{
			auto value = counter[i];
			if (value > 0)
			{
				probabilitySum += (double)value * log((double)value);
			}
		}

		delete[] counter;
		return static_cast<float>((log(N_d) - (1.0 / N_d) * probabilitySum) /log2(imageValueScale));
	}

	template <typename T>
	inline Tensor<int> quanticeImageValues(const Tensor<T> &image, int64_t nrValues)
	{
		auto max = image[0];
		auto min = image[0];
		for (size_t i = 1; i < image.GetTotalSize(); i++)
		{
			auto value = image[i];
			if (value > max)
				max = value;
			if (value < min)
				min = value;
		}
		float scaling = static_cast<float>(nrValues) / (max - min);

		auto channels = image.GetRank() == 2 ? 1 : image.GetDimension(2);
		Tensor<int> quanticed({image.GetDimension(0), image.GetDimension(1), channels});
		for (size_t i = 0; i < image.GetTotalSize(); i++)
		{
			quanticed[i] = static_cast<int>((image[i] - min) * scaling);
		}
		return quanticed;
	}
};