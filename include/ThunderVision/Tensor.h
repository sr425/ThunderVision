#pragma once

#include <vector>
#include <stdexcept>

//#define MEM_CHECK
#include "Exceptions.h"

#include <iostream>

namespace ThunderVision
{
template <typename T>
class Tensor
{
  public:
	Tensor() {}

	Tensor(std::vector<size_t> dimensions)
	{
		Resize(dimensions);
	}

	Tensor(std::initializer_list<size_t> dimensions)
	{
		Resize(std::vector<size_t>(dimensions));
	}

	void Resize(std::initializer_list<size_t> dimensions)
	{
		Resize(std::vector<size_t>(dimensions));
	}

	void Resize(std::vector<size_t> dimensions)
	{
		//std::cout << "Allocating tensor" << std::endl;
		_dimensions = dimensions;
		auto totalSize = computeTotalSize(_dimensions);
		_data.resize(totalSize);
		if (_data.size() != totalSize)
		{
			throw new ThunderException("Allocating enough memory failed: " + std::to_string(totalSize));
		}
		_totalSize = totalSize;
		updateStrides();
	}

	void Reshape(std::initializer_list<size_t> dimensions)
	{
		Reshape(std::vector<size_t>(dimensions));
	}

	void Reshape(std::vector<size_t> dimensions)
	{
		auto totalSize = computeTotalSize(_dimensions);
		if (totalSize != _totalSize)
		{
			throw new ThunderException("The overall size of the tensor may not be changed by reshape");
		}
		_dimensions = dimensions;
	}

	/*For size_t*/
	inline T &operator[](size_t index)
	{
#ifdef MEM_CHECK
		if (index >= _totalSize)
		{
			throw new std::out_of_range("The given index is outside the tensor range: " + index);
		}
#endif // MEM_CHECK
		return _data[index];
	}

	inline const T &operator[](size_t index) const
	{
#ifdef MEM_CHECK
		if (index >= _totalSize)
		{
			throw new std::out_of_range("The given index is outside the tensor range: " + index);
		}
#endif // MEM_CHECK
		return _data[index];
	}

	inline T &At(std::initializer_list<size_t> position)
	{
		return _data[computeIndex(position)];
	}

	inline const T &At(std::initializer_list<size_t> position) const
	{
		return _data[computeIndex(position)];
	}

	inline size_t GetDimension(size_t index) const
	{
#ifdef MEM_CHECK
		if (index >= _dimensions.size())
			throw new std::out_of_range("The given index is greater than the available number of dimensions: " + index);
#endif
		return _dimensions[index];
	}

	inline size_t GetRank() const
	{
		return _dimensions.size();
	}

	inline size_t GetTotalSize() const
	{
		return _totalSize;
	}

	void Fill(T value)
	{
		for (size_t i = 0; i < _totalSize; i++)
		{
			_data[i] = value;
		}
	}

	void Fill(const std::vector<T> &data)
	{
		for (size_t i = 0; i < _totalSize; i++)
		{
			_data[i] = data[i];
		}
	}

	T Min() const
	{
		if (_totalSize == 0)
			throw new ThunderException("The tensor has to have a size > 0");
		auto min = _data[0];
		for (const auto &el : _data)
		{
			if (min > el)
				min = el;
		}
		return min;
	}

	T Max() const
	{
		if (_totalSize == 0)
			throw new ThunderException("The tensor has to have a size > 0");
		auto max = _data[0];
		for (const auto &el : _data)
		{
			if (max < el)
				max = el;
		}
		return max;
	}

	template <typename TOut>
	Tensor<TOut> AsType()
	{
		Tensor<TOut> result(_dimensions);
		for (size_t i = 0; i < _totalSize; i++)
			result[i] = static_cast<TOut>(_data[i]);
		return result;
	}

	void Squeeze()
	{
		std::vector<size_t> newDimensions;
		for (const auto &dim : _dimensions)
		{
			if (dim != 1)
			{
				newDimensions.push_back(dim);
			}
		}
		_dimensions = newDimensions;
	}

  protected:
	std::vector<size_t> _dimensions;
	std::vector<T> _data;
	size_t _totalSize;

	std::vector<size_t> _dimensionStrides;

	inline size_t computeIndex(std::vector<size_t> position) const
	{
		if (position.size() > _dimensions.size())
		{
			throw new ThunderException("The given position has to be of the same dimensionality as the tensor");
		}

		size_t positionIndex = 0;
		for (size_t i = 0; i < position.size(); i++)
		{
			positionIndex += position[i] * _dimensionStrides[i];
		}
		return positionIndex;
	}

	inline size_t computeIndex(std::initializer_list<size_t> pos) const
	{
		auto position = std::vector<size_t>(pos);
		return computeIndex(position);
	}

	inline void updateStrides()
	{
		std::vector<size_t> strides(_dimensions.size());
		strides[_dimensions.size() - 1] = 1;
		for (int i = _dimensions.size() - 1; i > 0; i--)
		{
			strides[i - 1] = strides[i] * _dimensions[i];
		}
		_dimensionStrides = strides;
	}

  private:
	size_t computeTotalSize(std::vector<size_t> &dimensions)
	{
		size_t totalSize = 1;
		for (auto &size : dimensions)
		{
			totalSize *= size;
		}
		return totalSize;
	}
};
}; // namespace ThunderVision