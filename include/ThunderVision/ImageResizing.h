#pragma once
#include "Tensor.h"

#include <iostream>

namespace ThunderVision
{
	class ImageResizing
	{
	public:
		template<typename TIn, typename TOut> Tensor<TOut> DownscaleImage(const Tensor<TIn>& input, const size_t outputWidth, const size_t outputHeight)
		{
			if (input.GetRank() != 3 && input.GetRank() != 2)
				throw new ThunderException("Only images (tensors with two or three dimensions) are supported");
			if (outputWidth > input.GetDimension(1) || outputHeight > input.GetDimension(0))
				throw new ThunderException("The output size must be smaller than the input size");

			const size_t channels = input.GetRank() == 3 ? input.GetDimension(2) : 1;
			const int64_t width = static_cast<int64_t>(input.GetDimension(1));
			const int64_t height = static_cast<int64_t>(input.GetDimension(0));

			const float scaleX = static_cast<float>(width) / outputWidth;
			const float scaleY = static_cast<float>(height) / outputHeight;

			Tensor<TOut> result({ outputHeight, outputWidth, channels });

			size_t outIndex = 0;
			for (int64_t y_out = 0; y_out < outputHeight; y_out++)
			{
				for (int64_t x_out = 0; x_out < outputWidth; x_out++)
				{
					float y = scaleY * y_out;
					float x = scaleX * x_out;

					for (size_t c = 0; c < channels; c++, outIndex++)
					{
						result[outIndex] = static_cast<TOut>(GetValue(input, x, y, c));
					}
				}
			}
			return result;
		}

	private:
		template<typename TIn> inline float GetValue(const Tensor<TIn>& input, const float x, const float y, const size_t c)
		{
			int64_t x_p = (int64_t)x;
			int64_t y_p = (int64_t)y;

			const size_t channels = input.GetRank() == 2 ? 1 : input.GetDimension(2);

			const size_t width = input.GetDimension(1);
			size_t basePos = (y_p * width + x_p)* channels + c;
			auto pix00 = input[basePos];

			auto pix10 = pix00;
			auto pix01 = pix00;
			auto pix11 = pix00;

			if (x_p + 1 < input.GetDimension(1))
			{
				pix10 = input[basePos + channels];

				if (y_p + 1 < input.GetDimension(0))
				{
					pix01 = input[basePos + width * channels];
					pix11 = input[basePos + width * channels + channels];
				}
			}

			float e_x = x - x_p;
			float e_y = y - y_p;

			return (1.0f - e_x) * (1.0f - e_y) *pix00
				+ e_x * (1.0f - e_y) * pix10
				+ (1.0f - e_x) * e_y * pix01
				+ e_x * e_y*pix11;
		}
	};
}