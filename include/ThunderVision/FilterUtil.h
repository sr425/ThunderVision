#pragma once
#include <vector>

#include "Tensor.h"

namespace ThunderVision
{
class FilterUtil
{
  public:
	template <int X, int Y, typename TOut, typename TIn, typename TMask>
	Tensor<TOut> ApplyFilter(const Tensor<TIn> &input, const Tensor<TMask> &mask)
	{
		static_assert(X >= 0, "X has to be greater or equal to zero");
		static_assert(Y >= 0, "Y has to be greater or equal to zero");

		if (mask.GetRank() > 1)
			throw new ThunderException("The mask has to be 1D");
		if (mask.GetDimension(0) % 2 != 1)
			throw new ThunderException("The mask size has to be uneven");
		if (input.GetRank() != 3 && input.GetRank() != 2)
			throw new ThunderException("At the moment only 2D and 3D tensors are supported");

		const int64_t maskSize = static_cast<int64_t>(mask.GetDimension(0));
		const int64_t maskSizeHalf = (maskSize - 1) / 2;

		const int64_t height = static_cast<int64_t>(input.GetDimension(0));
		const int64_t width = static_cast<int64_t>(input.GetDimension(1));
		const int64_t channels = input.GetRank() == 3 ? static_cast<int64_t>(input.GetDimension(2)) : 1;

		Tensor<TOut> blurred({input.GetDimension(0), input.GetDimension(1), static_cast<size_t>(channels)});

		size_t pos = 0;
		for (int64_t y = 0; y < height; y++)
		{
			for (int64_t x = 0; x < width; x++)
			{
				for (size_t c = 0; c < channels; c++, pos++)
				{
					double value = 0.0;
					for (int64_t i = 0; i < maskSize; i++)
					{
						auto r_pos_i = i - maskSizeHalf;

						int offset_x = r_pos_i * X;
						int offset_y = r_pos_i * Y;

						auto target_x = x + offset_x;
						auto target_y = y + offset_y;

						if (X != 0)
						{
							if (target_x < 0)
							{
								target_x = std::abs(target_x + 1);
								offset_x = target_x - x;
							}
							else if (target_x >= width)
							{
								target_x = width + maskSizeHalf - i;
								offset_x = target_x - x;
							}
						}

						if (Y != 0)
						{
							if (target_y < 0)
							{
								target_y = std::abs(target_y + 1);
								offset_y = target_y - y;
							}
							else if (target_y >= height)
							{
								target_y = height + maskSizeHalf - i;
								offset_y = target_y - y;
							}
						}
						value += mask[i] * input[pos + static_cast<size_t>((offset_y * width + offset_x) * channels)];
					}
					blurred[pos] = static_cast<TOut>(value);
				}
			}
		}
		return blurred;
	}
};
} // namespace ThunderVision