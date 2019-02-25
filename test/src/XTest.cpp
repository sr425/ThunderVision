// XTest.cpp : Defines the entry point for the console application.
//
#define NOMINMAX
#include <iostream>
#include <chrono>
#include <string>

#include <ColorspaceConversion.h>
#include <SemiGlobalMatching.h>
#include <LaneDetection.h>
#include <MedianFilter.h>

#include "ImageLoader.h"
#include "ChannelConverter.h"
#include "TestMedian.h"
#include "Entropy.h"
#include "GaussianBlur.h"

using namespace ThunderVision;

int main()
{
	try
	{
		ColorspaceConversion conversion;

		std::cout << "Starting ..." << std::endl;
		auto leftImage = ImageLoader::LoadImage("path/to/left/img.png");
		auto rightImage = ImageLoader::LoadImage("path/to/right/img.png");

		auto startGrayscale = std::chrono::high_resolution_clock::now();
		leftImage = conversion.ConvertToGrayscale<uint8_t>(leftImage);
		leftImage.Squeeze();
		rightImage = conversion.ConvertToGrayscale<uint8_t>(rightImage);
		rightImage.Squeeze();
		auto endGrayscale = std::chrono::high_resolution_clock::now();
		std::cout << "Time for grayscale conversion (2x): " << std::chrono::duration_cast<std::chrono::milliseconds>(endGrayscale - startGrayscale).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(endGrayscale - startGrayscale).count() << " \xE6s)" << std::endl;

		SemiGlobalMatching sgm(64, false, AggregationDirections::Nr8);
		sgm.Prepare(leftImage.GetDimension(1), leftImage.GetDimension(0));
		std::cout << "SGM init finished" << std::endl;

		auto start = std::chrono::high_resolution_clock::now();
		auto disparities = sgm.ComputeDisparities(leftImage, rightImage);

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Total SGM time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms (" << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " \xE6s)" << std::endl;

		for (size_t i = 0; i < disparities.GetTotalSize(); i++)
		{
			if (disparities[i] > 100)
				disparities[i] = 0;
		}
		ImageLoader::SaveTensorScaled(disparities.AsType<float>(), "disp.png");

		std::cout << "------ Finished ------" << std::endl
				  << "Enter character and press enter key to exit" << std::endl;
		int dummy;
		//std::cin >> dummy;
		return 0;
	}
	catch (ThunderVision::ThunderException *e)
	{
		std::cout << "Caught exception " << e->getMessage() << std::endl;
		return -1;
	}
}
