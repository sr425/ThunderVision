#pragma once

#include <algorithm>
#include "ext/libpopcnt.h"
#include <cmath>

#include "Tensor.h"
#include "MedianFilter.h"
#include "Exceptions.h"

#define TIME_MEASUREMENT

#ifdef TIME_MEASUREMENT
#include <chrono>
#include <iostream>
#endif // TIME_MEASUREMENT

//#define OPEN_MP
#ifdef OPEN_MP
#include <omp.h>
#endif

namespace ThunderVision
{
enum class MatchingDirection
{
	lr,
	rl
};

enum class AggregationDirections
{
	Nr8,
	Nr4_Diag,
	Nr4_Axis
};

enum class CostFunction
{
	CENSUS
};

class SemiGlobalMatching
{
  public:
	SemiGlobalMatching(size_t maxDisparity, bool consistencyCheck, AggregationDirections nrAggregations);
	~SemiGlobalMatching();

	void Prepare(size_t width, size_t height);

	template <typename T>
	Tensor<float> ComputeDisparities(const Tensor<T> &leftImage, const Tensor<T> &rightImage)
	{
		if (leftImage.GetRank() != 2 || rightImage.GetRank() != 2)
		{
			throw new ThunderException("The input images have to be 2D grayscale images (rank 2 tensors).");
		}

		if (!prepared || leftImage.GetDimension(0) != censusLeft.GetDimension(0) || leftImage.GetDimension(1) != censusRight.GetDimension(1))
		{
			Prepare(leftImage.GetDimension(1), leftImage.GetDimension(0));
		}

#ifdef TIME_MEASUREMENT
		auto start_census = std::chrono::high_resolution_clock::now();
#endif
		ComputeCENSUSVectors<5, 5, uint64_t>(leftImage, censusLeft);
		ComputeCENSUSVectors<5, 5, uint64_t>(rightImage, censusRight);

#ifdef TIME_MEASUREMENT
		auto end_census = std::chrono::high_resolution_clock::now();
		std::cout << "Time for CENSUS computation (5x5): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_census - start_census).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(end_census - start_census).count() << " \xE6s)" << std::endl;
#endif

		if (!_consistencyCheck)
		{
			return ComputeMinimalMatchingCostImage<MatchingDirection::lr>(censusLeft, censusRight, _maxDisparity, true);
		}

#ifndef OPEN_MP
		auto matchingLeft = ComputeMinimalMatchingCostImage<MatchingDirection::lr>(censusLeft, censusRight, _maxDisparity, false);
		auto matchingRight = ComputeMinimalMatchingCostImage<MatchingDirection::rl>(censusLeft, censusRight, _maxDisparity, false);
		return LeftToRightConsistencyCheck(matchingLeft, matchingRight, 1.1f);
#else
		std::vector<Tensor<float>> matchings(2);
#pragma omp parallel num_threads(2)
#pragma omp for
		for (int i = 0; i < 2; i++)
		{
			if (i == 0)
			{
				auto matchingLeft = ComputeMinimalMatchingCostImage<MatchingDirection::lr>(censusLeft, censusRight, false);
				matchings[i] = matchingLeft;
			}
			else
			{
				auto matchingRight = ComputeMinimalMatchingCostImage<MatchingDirection::rl>(censusLeft, censusRight, false);
				matchings[i] = matchingRight;
			}
		}
		return LeftToRightConsistencyCheck(matchings[0], matchings[1], 1.1f);
#endif
	}

  private:
	size_t _maxDisparity;
	size_t _consistencyCheck;

	// Attention P2 >= P1 must hold
	// 4*P1 + 4*P2 < 255 (suggested by Daimler for some nice properties)
	const uint8_t P1 = 20;
	//15 original
	const uint8_t P2 = 40;
	//100 original
	const uint16_t errorPixelValue = static_cast<uint16_t>(UINT16_MAX - P2);
	const uint16_t invalid_census_value = UINT16_MAX;
	float btNormalizationValue = (UINT16_MAX - P2 - 10.0f) / UINT16_MAX;

	AggregationDirections _aggregationDirections;

	MedianFilter _medianFilter;

	//Class memory buffers
	bool prepared = false;
	Tensor<uint64_t> censusLeft;
	Tensor<uint64_t> censusRight;
	Tensor<unsigned int> aggregatedCosts;
	Tensor<unsigned int> tempBuffer;
	Tensor<uint16_t> costVolume;
	Tensor<float> minimalDisparities;

	/* Methods implemented in .cpp*/
	void AggregateCosts(const Tensor<uint16_t> &costVolume, const bool internalParallel, Tensor<unsigned int> &aggregatedCosts, Tensor<unsigned int> &tempBuffer);

	inline void AggregatePositionCost(const Tensor<uint16_t> &costVolume, Tensor<unsigned int> &aggregatedCosts, Tensor<unsigned int> &temp, const int64_t maxDisp, const int64_t pos, const int64_t direction);
	inline void CopyCostsToAggregation(const Tensor<uint16_t> &costVolume, Tensor<unsigned int> &aggregatedCosts, Tensor<unsigned int> &temp, const int64_t maxDisp, const int64_t pos);
	/**
	* X>0 means iteration is applied from left to right, Y > 0 means iteration is applied from top to bottom. X=0 or Y=0 means path is not applied in this direction.
	*/
	template <int X, int Y>
	void AggregateCosts(const Tensor<uint16_t> &costVolume, Tensor<unsigned int> &aggregatedCosts, Tensor<unsigned int> &tempBuffer);

	void ComputeMinimalDisparity(const Tensor<unsigned int> &costVolume, Tensor<float> &minimalDisparities);

	Tensor<float> LeftToRightConsistencyCheck(const Tensor<float> &leftDisparityImage, const Tensor<float> &rightDisparityImage, const float epsilon);
	/******************************/

	template <MatchingDirection direction>
	Tensor<float> ComputeMinimalMatchingCostImage(const Tensor<uint64_t> &censusLeft, const Tensor<uint64_t> &censusRight, const size_t maxDisp, bool internalParallel)
	{
#ifdef TIME_MEASUREMENT
		auto start_cost_volume = std::chrono::high_resolution_clock::now();
#endif

		ComputeMatchingCostsCENSUS<direction>(censusLeft, censusRight, maxDisp, costVolume);

#ifdef TIME_MEASUREMENT
		auto end_cost_volume = std::chrono::high_resolution_clock::now();
		std::cout << "Time for cost volume computation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cost_volume - start_cost_volume).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(end_cost_volume - start_cost_volume).count() << " \xE6s)" << std::endl;
		auto start_aggregation = std::chrono::high_resolution_clock::now();
#endif

		AggregateCosts(costVolume, internalParallel, aggregatedCosts, tempBuffer);

#ifdef TIME_MEASUREMENT
		auto end_aggregation = std::chrono::high_resolution_clock::now();
		std::cout << "Time for cost aggregation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_aggregation - start_aggregation).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(end_aggregation - start_aggregation).count() << " \xE6s)" << std::endl;
		auto start_minimal_comp = std::chrono::high_resolution_clock::now();
#endif

		ComputeMinimalDisparity(aggregatedCosts, minimalDisparities);

#ifdef TIME_MEASUREMENT
		auto end_minimal_comp = std::chrono::high_resolution_clock::now();
		std::cout << "Time for cost minimization: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_minimal_comp - start_minimal_comp).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(end_minimal_comp - start_minimal_comp).count() << " \xE6s)" << std::endl;
		auto start_median = std::chrono::high_resolution_clock::now();
#endif

		auto l_minimalDisparities = _medianFilter.ApplyMedianFilter<3, 3>(minimalDisparities);

#ifdef TIME_MEASUREMENT
		auto end_median = std::chrono::high_resolution_clock::now();
		std::cout << "Time for median computation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_median - start_median).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(end_median - start_median).count() << " \xE6s)" << std::endl;
#endif
		return l_minimalDisparities;
	}

	template <uint32_t filtersize_x, uint32_t filtersize_y, typename TVector, typename T>
	void ComputeCENSUSVectors(const Tensor<T> &image, Tensor<TVector> &censusImage)
	{
		static_assert(filtersize_x % 2 == 1, "The CENSUS mask width has to be uneven.");
		static_assert(filtersize_y % 2 == 1, "The CENSUS mask height has to be uneven.");

		const size_t width = image.GetDimension(1);

		constexpr size_t size_x_h = (filtersize_x - 1) / 2;
		constexpr size_t size_y_h = (filtersize_y - 1) / 2;

		const size_t endIndex_y = image.GetDimension(0) - size_y_h;
		const size_t endIndex_x = width - size_x_h;

		censusImage.Fill(invalid_census_value);
		for (size_t y = size_y_h; y < endIndex_y; y++)
		{
			for (size_t x = size_x_h; x < endIndex_x; x++)
			{
				size_t pos = y * width + x;
				censusImage[pos] = ComputeCENSUSVector<filtersize_x, filtersize_y, TVector>(image, pos);
			}
		}
	}

	template <uint32_t filtersize_x, uint32_t filtersize_y, typename TVector, typename T>
	TVector ComputeCENSUSVector(const Tensor<T> &image, size_t pos)
	{
		static_assert(filtersize_x % 2 == 1, "The CENSUS mask width has to be uneven.");
		static_assert(filtersize_y % 2 == 1, "The CENSUS mask width has to be uneven.");

		const T basePixel = image[pos];
		const size_t width = image.GetDimension(1);

		constexpr size_t size_x_h = (filtersize_x - 1) / 2;
		constexpr size_t size_y_h = (filtersize_y - 1) / 2;

		//Compute base position as beeing half y over pos and half x to the left
		size_t basePos = pos - (size_y_h * width) - size_x_h;

		size_t x_t;
		TVector vector = 0;
		TVector posEncoding = 1;
		for (size_t y_t = 0; y_t < filtersize_y; y_t++)
		{
			for (x_t = 0; x_t < filtersize_x; x_t++)
			{
				//Write a one to the bit vector if current pixel is large than the base pixel
				if (basePixel < image[basePos + x_t])
				{
					vector |= posEncoding;
				}
				//Shift the bit vector to new position
				posEncoding <<= 1;
			}
			basePos += width;
		}
		return vector;
	}

	inline uint16_t computeHammingDistance(const uint64_t x1, const uint64_t x2)
	{
		return static_cast<uint16_t>(popcnt64(x1 ^ x2));
	}

	template <MatchingDirection direction>
	void ComputeMatchingCostsCENSUS(const Tensor<uint64_t> &censusLeft, const Tensor<uint64_t> &censusRight, const size_t maxDisp, Tensor<uint16_t> &costVolume)
	{
		const size_t width = censusLeft.GetDimension(1);
		const size_t height = censusLeft.GetDimension(0);

		uint64_t baseVector;
		size_t x, d = 0;
		size_t pos = 0;
		size_t costPos = 0;

		costVolume.Fill(errorPixelValue);
		for (size_t y = 0; y < height; y++)
		{
			for (x = 0; x < width; x++, pos++, costPos += maxDisp)
			{
				if ((direction == MatchingDirection::lr) ? censusLeft[pos] == invalid_census_value : censusRight[pos] == invalid_census_value)
				{
					continue;
				}

				baseVector = (direction == MatchingDirection::lr) ? censusLeft[pos] : censusRight[pos];
				for (d = 0; d < maxDisp && (direction == MatchingDirection::lr ? x >= d : x + d < width); d++)
				{
					if (direction == MatchingDirection::lr)
					{
						costVolume[costPos + d] = computeHammingDistance(baseVector, censusRight[pos - d]);
					}
					else
					{
						costVolume[costPos + d] = computeHammingDistance(baseVector, censusLeft[pos + d]);
					}
				}
			}
		}
	}
};
} // namespace ThunderVision
