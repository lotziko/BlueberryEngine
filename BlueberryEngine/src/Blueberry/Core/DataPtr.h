#pragma once

class Data;

namespace Blueberry
{
	template<class DataType>
	class DataPtr
	{
	public:
		DataPtr() = default;
		DataPtr(DataType* data);
		DataPtr(void* data);
		void operator=(Data* data);
		void operator=(void* data);
		DataType* operator->();

		DataType* Get() const;

	private:
		DataType* m_Data;
	};

	template<class DataType>
	inline DataPtr<DataType>::DataPtr(DataType* data)
	{
		static_assert(std::is_base_of<Data, DataType>::value, "Type is not derived from Data.");
		m_Data = data;
	}

	template<class DataType>
	inline DataPtr<DataType>::DataPtr(void* data)
	{
		m_Data = static_cast<DataType*>(data);
	}

	template<class DataType>
	inline void DataPtr<DataType>::operator=(Data* data)
	{
		static_assert(std::is_base_of<Data, DataType>::value, "Type is not derived from Data.");
		m_Data = static_cast<DataType*>(data);
	}

	template<class DataType>
	inline void DataPtr<DataType>::operator=(void* data)
	{
		m_Data = static_cast<DataType*>(data);
	}

	template<class DataType>
	inline DataType* DataPtr<DataType>::operator->()
	{
		return DataPtr<DataType>::Get();
	}

	template<class DataType>
	inline DataType* DataPtr<DataType>::Get() const
	{
		return m_Data;
	}
}