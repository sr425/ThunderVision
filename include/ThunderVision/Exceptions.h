#pragma once

#include <exception>
#include <string>

namespace ThunderVision
{

	class ThunderException : public std::exception
	{
	public:
		ThunderException(std::string message)
		{
			_message = message;
		}

		std::string getMessage()
		{
			return _message;
		}

	protected:
		std::string _message;
	};
}