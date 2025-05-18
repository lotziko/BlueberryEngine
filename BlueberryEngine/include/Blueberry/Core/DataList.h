#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	class Data;

	class DataListBase
	{
	public:
		virtual size_t Size() = 0;
		virtual void* EmplaceBack() = 0;
		virtual void* Get(const uint32_t& index) = 0;
	};

	template<class DataType>
	class DataList : public List<DataType>, public DataListBase
	{
	public:
		virtual size_t Size() final;
		virtual void* EmplaceBack() final;
		virtual void* Get(const uint32_t& index) final;
	};

	template<class DataType>
	inline size_t DataList<DataType>::Size()
	{
		return size();
	}

	template<class DataType>
	inline void* DataList<DataType>::EmplaceBack()
	{
		emplace_back();
		return data() + (size() - 1);
	}

	template<class DataType>
	inline void* DataList<DataType>::Get(const uint32_t& index)
	{
		return data() + index;
	}
}