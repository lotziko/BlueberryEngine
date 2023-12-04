#include "bbpch.h"
#include "Variant.h"

namespace Blueberry
{
	Variant::Variant(const int &$int)
	{
		m_IntData = $int;
	}

	Variant::Variant(const std::string& string)
	{
		m_StringData = string;
	}

	Variant::Variant(const Vector3& vector3)
	{
		m_Vector3Data = vector3;
	}

	Variant::Variant(const Quaternion& quaternion)
	{
		m_QuaternionData = quaternion;
	}

	Variant::Variant(Object* object)
	{
		m_ObjectData = object;
	}

	Variant::operator const int&()
	{
		return m_IntData;
	}

	Variant::operator const std::string&()
	{
		return m_StringData;
	}

	Variant::operator const Vector3&()
	{
		return m_Vector3Data;
	}

	Variant::operator const Quaternion&()
	{
		return m_QuaternionData;
	}

	Variant::operator Object*()
	{
		return m_ObjectData;
	}
}
