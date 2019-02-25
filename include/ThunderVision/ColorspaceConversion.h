#pragma once

#include "Tensor.h"

namespace ThunderVision
{
class ColorspaceConversion
{
  public:
	template <typename TOut, typename TIn>
	static Tensor<TOut> ConvertToGrayscale(const Tensor<TIn> &input)
	{
		if (input.GetRank() != 3 && input.GetDimension(2) != 3)
			throw new ThunderException("Conversion of images with rank < 3 and channels != 3 to grayscale is not supported. Only RGB tensors (tensors with 3 channels) are supported.");
		const size_t channels = 3;
		const size_t width = input.GetDimension(1);
		const size_t height = input.GetDimension(0);
		Tensor<TOut> result({height, width, 1});
		for (size_t i = 0, in_pos = 0; i < width * height; i++, in_pos += channels)
		{
			result[i] = static_cast<TOut>((input[in_pos] + input[in_pos + 1] + input[in_pos + 2]) / 3.0f);
		}
		return result;
	}
};
} // namespace ThunderVision