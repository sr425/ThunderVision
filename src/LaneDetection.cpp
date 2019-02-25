#include "LaneDetection.h"

#include <cassert>
#include <algorithm>

ThunderVision::LaneDetection::LaneDetection()
{
}


ThunderVision::LaneDetection::~LaneDetection()
{
}

ThunderVision::Tensor<int> ThunderVision::LaneDetection::DetectLaneCenter(const Tensor<int>& image)
{
	assert(image.GetRank() == 2);

	const int64_t width = static_cast<int64_t>(image.GetDimension(1));
	const int64_t totalSize = static_cast<int64_t>(image.GetTotalSize());

	Tensor<int> result({ image.GetDimension(0), image.GetDimension(1), image.GetDimension(2) });

	for (int64_t i = 0; i < totalSize; i++)
	{
		result[i] = 0;
		int64_t x = i % width;
		if (x < 1 || x >= width - 1)
			continue;

		if (image[i] != road)
			continue;

		if (!containsOtherValue(image, i, road))
			continue;

		result[i] = 255;
	}
	
	return result;
}

inline bool ThunderVision::LaneDetection::containsOtherValue(const Tensor<int>& image, int64_t pos, int value)
{
	const int64_t width = static_cast<int64_t>(image.GetDimension(1));
	const int64_t totalSize = static_cast<int64_t>(image.GetTotalSize());

	int64_t basePos = pos - width - 1;
	for (int64_t y = 0; y < 3; y++)
	{
		for (int64_t x = 0; x < 3; x++)
		{
			auto lookupPos = basePos + x;
			if (lookupPos < 0 || lookupPos >= totalSize)
				continue;

			if (image[lookupPos] != value)
				return true;
		}
		basePos += width;
	}
	return false;
}