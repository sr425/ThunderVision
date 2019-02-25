#pragma once
#define NOMINMAX
#include <cassert>
#include <algorithm>

#include "Tensor.h"

namespace ThunderVision
{
	class MedianFilter
	{
	public:
		MedianFilter() {  }
		~MedianFilter() {}

		template<size_t filtersize_x, size_t filtersize_y, typename T> Tensor<T> ApplyMedianFilter(const Tensor<T>& input)
		{
			static_assert(filtersize_x % 2 == 1, "Median filtersize has to be uneven.");
			static_assert(filtersize_y % 2 == 1, "Median filtersize has to be uneven.");
			assert(input.GetRank() == 2 || input.GetRank()==3);

			const size_t height = input.GetDimension(0);
			const size_t width = input.GetDimension(1);
			const size_t channels = input.GetRank() == 2 ? 1 : input.GetDimension(2);
	
			const int width_i = static_cast<int>(width);

			const size_t filtersize_x_h = (filtersize_x - 1) / 2;
			const size_t filtersize_y_h = (filtersize_y - 1) / 2;

			std::vector<T> mask(filtersize_x * filtersize_y);
			Tensor<T> result({ height, width, channels });

			auto error_value = std::numeric_limits<T>::max() - 1;//TODO: std::min(std::numeric_limits<T>::max(), std::abs(std::numeric_limits<T>::min()));

			for (size_t c = 0; c < channels; c++)
			{
				int pos = c;
				for (size_t y = 0; y < height; y++)
				{
					for (size_t x = 0; x < width; x++, pos+=channels)
					{
						int basePos = pos - (static_cast<int>(filtersize_y_h * width) + static_cast<int>(filtersize_x_h)) * channels;
						size_t targetPos = 0;

						for (size_t y_t = 0; y_t < filtersize_x; y_t++)
						{
							for (size_t x_t = 0; x_t < filtersize_y; x_t++, targetPos++)
							{
								if (x + x_t >= filtersize_x_h && y + y_t >= filtersize_y_h && x + x_t < width && y + y_t < height)
								{
									mask[targetPos] = input[basePos + x_t*channels];
								}
								else
								{
									mask[targetPos] = error_value;
									error_value *= -1;
								}
							}
							basePos += width_i * channels;
						}


						result[pos] = GetMedian(mask);
					}
				}
			}
			return result;
		}

	private:
		template<typename T> inline T GetMedian(std::vector<T>& mask)
		{
			std::sort(mask.begin(), mask.end());
			return mask[(mask.size() - 1) / 2];
		}
	};
}