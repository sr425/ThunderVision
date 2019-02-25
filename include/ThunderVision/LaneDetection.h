#pragma once

#include "Tensor.h"

namespace ThunderVision
{
	class LaneDetection
	{
	public:
		LaneDetection();
		~LaneDetection();

		Tensor<int> DetectLaneCenter(const Tensor<int>& image);
		
	private: 
		const int road = 0;
		const int sidewalk = 1;

		inline bool containsOtherValue(const Tensor<int>& image, int64_t pos, int value);
	};
}
