#pragma once

namespace Blueberry
{
	class Variant
	{
	public:
		Variant() = default;
		Variant(const int& $int);
		Variant(const std::string& string);
		Variant(const Vector3& vector3);
		Variant(const Quaternion& quaternion);
		Variant(Object* object);

		operator const int&();
		operator const std::string&();
		operator const Vector3&();
		operator const Quaternion&();
		operator Object*();

	private:
		int m_IntData;
		std::string m_StringData;
		Vector3 m_Vector3Data;
		Quaternion m_QuaternionData;
		Object* m_ObjectData;
	};
}