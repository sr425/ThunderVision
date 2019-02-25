#include "SemiGlobalMatching.h"

ThunderVision::SemiGlobalMatching::SemiGlobalMatching(size_t maxDisparity, bool consistencyCheck, AggregationDirections dirs)
{
	_aggregationDirections = dirs;
	_maxDisparity = maxDisparity;
	_consistencyCheck = consistencyCheck;

	static_assert(std::numeric_limits<int>::max() >= std::numeric_limits<int32_t>::max(), "The integer type on this system is too small to represent all of the asserted images properly.");
	static_assert(std::numeric_limits<unsigned int>::max() >= std::numeric_limits<uint32_t>::max(), "The integer type on this system is too small to represent all of the asserted images properly.");
}

ThunderVision::SemiGlobalMatching::~SemiGlobalMatching()
{
}

void ThunderVision::SemiGlobalMatching::Prepare(size_t width, size_t height)
{
	Tensor<unsigned int> l_aggregatedCosts({height, width, _maxDisparity});
	Tensor<unsigned int> l_tempBuffer({height, width, _maxDisparity});
	Tensor<uint16_t> l_costVolume({height, width, _maxDisparity});
	Tensor<float> l_minimalDisparities({height, width});
	Tensor<uint64_t> l_censusLeftImage({height, width});
	Tensor<uint64_t> l_censusRightImage({height, width});

	censusLeft = l_censusLeftImage;
	censusRight = l_censusRightImage;
	aggregatedCosts = l_aggregatedCosts;
	tempBuffer = l_tempBuffer;
	costVolume = l_costVolume;
	minimalDisparities = l_minimalDisparities;

	prepared = true;
}

void ThunderVision::SemiGlobalMatching::AggregateCosts(const Tensor<uint16_t> &costVolume, const bool internalParallel, Tensor<unsigned int> &aggregatedCosts, Tensor<unsigned int> &tempBuffer)
{
	aggregatedCosts.Fill(0);
	if (_aggregationDirections == AggregationDirections::Nr4_Diag)
	{
		AggregateCosts<1, 1>(costVolume, aggregatedCosts, tempBuffer);
		AggregateCosts<-1, 1>(costVolume, aggregatedCosts, tempBuffer);
		AggregateCosts<-1, -1>(costVolume, aggregatedCosts, tempBuffer);
		AggregateCosts<1, -1>(costVolume, aggregatedCosts, tempBuffer);
		return;
	}

	if (_aggregationDirections == AggregationDirections::Nr4_Axis)
	{
		AggregateCosts<1, 0>(costVolume, aggregatedCosts, tempBuffer);
		AggregateCosts<0, 1>(costVolume, aggregatedCosts, tempBuffer);
		AggregateCosts<-1, 0>(costVolume, aggregatedCosts, tempBuffer);
		AggregateCosts<0, -1>(costVolume, aggregatedCosts, tempBuffer);
		return;
	}

	AggregateCosts<1, 0>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<1, 1>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<0, 1>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<-1, 1>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<-1, 0>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<-1, -1>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<0, -1>(costVolume, aggregatedCosts, tempBuffer);
	AggregateCosts<1, -1>(costVolume, aggregatedCosts, tempBuffer);
}

inline void ThunderVision::SemiGlobalMatching::AggregatePositionCost(const Tensor<uint16_t> &costVolume,
																	 Tensor<unsigned int> &aggregatedCosts,
																	 Tensor<unsigned int> &temp,
																	 const int64_t maxDisp,
																	 const int64_t pos,
																	 const int64_t direction)
{
	auto posDir = pos + direction;
	auto pos_index = pos;
	auto minCosts = temp[posDir];
	const size_t e = posDir + maxDisp;
	for (size_t j = posDir; j < e; j++)
	{
		minCosts = std::min(temp[j], minCosts);
	}

	auto value = costVolume[pos] + std::min(temp[posDir], std::min(temp[posDir + 1] + P1, minCosts + P2)) - minCosts;
	temp[pos_index] = value;
	aggregatedCosts[pos_index] += value;

	const int64_t end = maxDisp - 1;
	for (int64_t d = 1; d < end; d++)
	{
		value = costVolume[pos + d] + std::min(temp[posDir + d], std::min(minCosts + P2, std::min(temp[posDir + d - 1], temp[posDir + d + 1]) + P1));
		value -= minCosts;

		temp[pos_index + d] = value;
		aggregatedCosts[pos_index + d] += value;
	}

	value = costVolume[pos + maxDisp - 1] + std::min(temp[posDir + maxDisp - 1], std::min(temp[posDir + maxDisp - 2] + P1, minCosts + P2));
	value -= minCosts;
	temp[pos_index + maxDisp - 1] = value;
	aggregatedCosts[pos_index + maxDisp - 1] += value;
}

inline void ThunderVision::SemiGlobalMatching::CopyCostsToAggregation(const Tensor<uint16_t> &costVolume,
																	  Tensor<unsigned int> &aggregatedCosts,
																	  Tensor<unsigned int> &temp,
																	  const int64_t maxDisp,
																	  const int64_t pos)
{
	for (int64_t d = 0; d < maxDisp; d++)
	{
		temp[pos + d] = costVolume[pos + d];
		aggregatedCosts[pos + d] += costVolume[pos + d];
	}
}

template <int X, int Y>
void ThunderVision::SemiGlobalMatching::AggregateCosts(const Tensor<uint16_t> &costVolume, Tensor<unsigned int> &aggregatedCosts, Tensor<unsigned int> &tempBuffer)
{
	const int64_t width = static_cast<int64_t>(costVolume.GetDimension(1));
	const int64_t height = static_cast<int64_t>(costVolume.GetDimension(0));
	const int64_t size = width * height;

	const int64_t maxDisp = static_cast<int64_t>(costVolume.GetDimension(2));

	const bool startTop = (Y > 0 || (Y == 0 && X == 1));
	for (int64_t i = startTop ? 0 : size - 1; startTop ? i < size : i >= 0; startTop ? i++ : i--)
	{
		int64_t pos = i * maxDisp;

		int64_t x = i % width;
		if ((X > 0 && x == 0) || (X < 0 && x == width - 1))
		{
			CopyCostsToAggregation(costVolume, aggregatedCosts, tempBuffer, maxDisp, pos);
			continue;
		}

		int64_t y = (i - x) / width;
		if ((Y > 0 && y == 0) || (Y < 0 && y == height - 1))
		{
			CopyCostsToAggregation(costVolume, aggregatedCosts, tempBuffer, maxDisp, pos);
			continue;
		}

		int64_t direction = ((-1 * X) + (-1 * Y * width)) * maxDisp;
		AggregatePositionCost(costVolume, aggregatedCosts, tempBuffer, maxDisp, pos, direction);
	}
}

void ThunderVision::SemiGlobalMatching::ComputeMinimalDisparity(const Tensor<unsigned int> &aggregatedCosts, Tensor<float> &minimalDisparities)
{
	const size_t height = aggregatedCosts.GetDimension(0);
	const size_t width = aggregatedCosts.GetDimension(1);
	const size_t maxDisp = aggregatedCosts.GetDimension(2);

	size_t d = 0;
	for (size_t i = 0, cost_pos = 0; i < width * height; i++, cost_pos += maxDisp)
	{
		unsigned int minCost = std::numeric_limits<unsigned int>::max();
		size_t minDisparity = 0;
		for (d = 0; d < maxDisp; d++)
		{
			auto value = aggregatedCosts[cost_pos + d];

			if (value < minCost)
			{
				minCost = value;
				minDisparity = d;
			}
		}

		minimalDisparities[i] = static_cast<float>(minDisparity);
	}
}

ThunderVision::Tensor<float> ThunderVision::SemiGlobalMatching::LeftToRightConsistencyCheck(const Tensor<float> &leftDisparityImage, const Tensor<float> &rightDisparityImage, const float epsilon)
{
	float disparityValueLeft, disparityValueRight;

	const int64_t width = static_cast<int64_t>(leftDisparityImage.GetDimension(1));
	const int64_t height = static_cast<int64_t>(leftDisparityImage.GetDimension(0));

	Tensor<float> consistencyCheckedImage({leftDisparityImage.GetDimension(0), leftDisparityImage.GetDimension(1)});

	int64_t posLeft = 0;
	int64_t x, posRight;
	for (int64_t y = 0; y < height; y++)
	{
		for (x = 0; x < width; x++, posLeft++)
		{
			disparityValueLeft = leftDisparityImage[posLeft];

			int new_x = static_cast<int>(static_cast<float>(x) - disparityValueLeft);
			posRight = posLeft + (new_x - x);

			if (new_x >= 0)
				disparityValueRight = rightDisparityImage[posRight];
			else
			{
				disparityValueRight = std::numeric_limits<float>::max();
			}

			if (std::abs(disparityValueLeft - disparityValueRight) < epsilon)
			{
				consistencyCheckedImage[posLeft] = (disparityValueLeft + disparityValueRight) / 2.0f;
			}
			else
			{
				consistencyCheckedImage[posLeft] = std::numeric_limits<float>::max();
			}
		}
	}
	return consistencyCheckedImage;
}
