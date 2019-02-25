#pragma once

#include <cstdint>
#include <string>
#include <cassert>
#include <algorithm>
#include <iterator>
#include <iostream>

#include <Tensor.h>

#include "lodepng.h"

using namespace ThunderVision;

class ImageLoader
{
  public:
	ImageLoader();
	~ImageLoader();

	static Tensor<uint8_t> LoadImage(std::string path)
	{
		std::vector<unsigned char> image;
		unsigned width, height;

		auto error = lodepng::decode(image, width, height, path.c_str());
		if (error)
		{
			throw new ThunderException("Image loading failed: " + std::string(lodepng_error_text(error)));
		}

		Tensor<uint8_t> result({height, width, 3});
		size_t resCnt = 0;
		for (size_t i = 0; i < width * height * 4; i += 4)
		{
			for (size_t j = 0; j < 3; j++)
			{
				result[resCnt] = image[i + j];
				resCnt++;
			}
		}
		return result;
	}

	static void SaveTensor(const Tensor<int> &tensor, std::string path)
	{
		const size_t channels = tensor.GetRank() == 2 ? 1 : tensor.GetDimension(2);
		Tensor<uint8_t> result({tensor.GetDimension(0), tensor.GetDimension(1), channels});
		for (size_t i = 0; i < tensor.GetTotalSize(); i++)
			result[i] = static_cast<uint8_t>(std::min(255, std::max(0, tensor[i])));
		SaveTensor(result, path);
	}

	static void SaveTensor(const Tensor<uint8_t> &tensor, std::string path)
	{
		assert((tensor.GetRank() == 3 && (tensor.GetDimension(2) == 3 || tensor.GetDimension(2) == 1)) || tensor.GetRank() == 2);

		bool singleChannel = (tensor.GetRank() == 2 || tensor.GetDimension(2) == 1);

		auto width = tensor.GetDimension(1);
		auto height = tensor.GetDimension(0);
		std::vector<unsigned char> image(height * width * 4);

		size_t resCnt = 0;
		for (size_t i = 0; i < width * height * 4; i += 4)
		{
			if (singleChannel)
			{
				for (size_t j = 0; j < 3; j++)
				{
					image[i + j] = tensor[resCnt];
				}
				resCnt++;
			}
			else
			{
				for (size_t j = 0; j < 3; j++)
				{
					image[i + j] = tensor[resCnt];
					resCnt++;
				}
			}
			image[i + 3] = 255;
		}

		auto error = lodepng::encode(path.c_str(), image, static_cast<unsigned>(width), static_cast<unsigned>(height));
		if (error)
			throw new ThunderException("Tensor saving failed" + std::string(lodepng_error_text(error)));
	}

	static void SaveTensorScaled(const Tensor<float> &tensor, std::string path)
	{
		assert((tensor.GetRank() == 3 && tensor.GetDimension(2) == 1) || tensor.GetRank() == 2);

		auto width = tensor.GetDimension(1);
		auto height = tensor.GetDimension(0);

		Tensor<uint8_t> scaled({height, width});
		auto min = tensor.Min();
		auto max = tensor.Max();
		auto scale = 255.0f / (max - min);

		for (size_t i = 0; i < width * height; i++)
		{
			scaled[i] = static_cast<uint8_t>(tensor[i] * scale);
		}
		SaveTensor(scaled, path);
	}
};
