#pragma once
#include <Tensor.h>

using namespace ThunderVision;

class TestMedian
{
public:
	TestMedian() {}
	~TestMedian() {}

	template<typename T> Tensor<T> smooth_along_last_2_axis(const Tensor<T> &input, int kernelsize)
	{
		if (kernelsize % 2 != 1)
			throw new ThunderException("The kernel size has to be uneven");
		auto size_x = kernelsize;
		auto size_y = kernelsize;

		const size_t height = input.GetDimension(1);
		const size_t width = input.GetDimension(2);
		const int width_i = static_cast<int>(width);

		const size_t size_x_h = (size_x - 1) / 2;
		const size_t size_y_h = (size_y - 1) / 2;

		std::vector<T> mask(size_x * size_y);
		Tensor<T> result({ input.GetDimension(0), height, width });

		auto error_value = std::numeric_limits<T>::max() - 1; //TODO std::min(std::numeric_limits<T>::max(), std::abs(std::numeric_limits<T>::min()));

		int pos = 0;
		for (size_t c = 0; c < input.GetDimension(0); c++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t x = 0; x < width; x++, pos++)
				{
					int basePos = pos - static_cast<int>(size_y_h * width) - static_cast<int>(size_x_h);
					size_t targetPos = 0;

					for (size_t y_t = 0; y_t < size_x; y_t++)
					{
						for (size_t x_t = 0; x_t < size_y; x_t++, targetPos++)
						{
							if (x + x_t >= size_x_h && y + y_t >= size_y_h && x + x_t < width && y + y_t < height)
							{
								mask[targetPos] = input[basePos + x_t];
							}
							else
							{
								mask[targetPos] = error_value;
								error_value *= -1;
							}
						}
						basePos += width_i;
					}

					result[pos] = GetMedian(mask);
				}
			}
		}
		return result;
	}

	private:
		template<typename T> inline T GetMedian(std::vector<T> &mask)
		{
			std::sort(mask.begin(), mask.end());
			return mask[(mask.size() - 1) / 2];
		}
};