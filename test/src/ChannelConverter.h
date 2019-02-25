#pragma once
#include <Tensor.h>

using namespace ThunderVision;

class ChannelConverter
{
public:
	ChannelConverter();
	~ChannelConverter();

	template<typename T> Tensor<T> HWC_to_CWH(const Tensor<T>& input)
	{
		auto height = input.GetDimension(0);
		auto width = input.GetDimension(1);
		auto channels = input.GetDimension(2);

		Tensor<T> result({channels, width, height });

		auto out_stride = width * height;
		auto in_stride = width * channels;		

		for (size_t c = 0; c < channels; c++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++)
				{
					result[c * out_stride + x * height + y] = input[y * in_stride + x * channels + c];
				}
			}
		}
		return result;
	}

	template<typename T> Tensor<T> HWC_to_CHW(const Tensor<T>& input)
	{
		auto height = input.GetDimension(0);
		auto width = input.GetDimension(1);
		auto channels = input.GetDimension(2);

		Tensor<T> result({ channels, height, width });

		auto out_stride = width * height;
		auto in_stride = width * channels;

		for (size_t c = 0; c < channels; c++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++)
				{
					result[c * out_stride + y * width + x] = input[y * in_stride + x * channels + c];
				}
			}
		}
		return result;
	}

	template<typename T> Tensor<T> CWH_to_HWC(const Tensor<T>& input)
	{
		auto height = input.GetDimension(2);
		auto width = input.GetDimension(1);
		auto channels = input.GetDimension(0);

		Tensor<T> result({ height, width, channels});

		auto out_stride = width * channels;
		auto in_stride = width * height;

		for (size_t c = 0; c < channels; c++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++)
				{
					result[y * out_stride + x * channels + c] = input[c * in_stride + x * height + y];
				}
			}
		}
		return result;
	}

	template<typename T> Tensor<T> CHW_to_HWC(const Tensor<T>& input)
	{
		auto height = input.GetDimension(1);
		auto width = input.GetDimension(2);
		auto channels = input.GetDimension(0);

		Tensor<T> result({ height, width, channels });

		auto out_stride = width * channels;
		auto in_stride = width * height;

		for (size_t c = 0; c < channels; c++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++)
				{
					result[y * out_stride + x * channels + c] = input[c * in_stride + y * width + x];
				}
			}
		}
		return result;
	}
};

