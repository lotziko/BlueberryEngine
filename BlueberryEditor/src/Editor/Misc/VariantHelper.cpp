#include "VariantHelper.h"

namespace Blueberry
{
	BindingType VariantHelper::GetChildType(const BindingType& type)
	{
		switch (type)
		{
		case BindingType::IntList:
			return BindingType::Int;
		case BindingType::FloatList:
			return BindingType::Float;
		case BindingType::StringList:
			return BindingType::String;
		case BindingType::Vector2List:
			return BindingType::Vector2;
		case BindingType::Vector3List:
			return BindingType::Vector3;
		case BindingType::Vector4List:
			return BindingType::Vector4;
		case BindingType::ObjectPtrList:
			return BindingType::ObjectPtr;
		case BindingType::DataList:
			return BindingType::Data;
		default:
			return BindingType::None;
		}
	}

	void VariantHelper::GetDefaultValue(const BindingType& type, Variant& value)
	{
		switch (type)
		{
		case BindingType::Bool:
			value = false;
			break;
		case BindingType::Int:
			value = 0;
			break;
		case BindingType::Uint:
			value = 0u;
			break;
		case BindingType::Float:
			value = 0.0f;
			break;
		case BindingType::Enum:
			value = 0;
			break;
		case BindingType::String:
			value = String();
			break;
		case BindingType::Vector2:
			value = Vector2::Zero;
			break;
		case BindingType::Vector3:
			value = Vector3::Zero;
			break;
		case BindingType::Vector4:
			value = Vector4::Zero;
			break;
		case BindingType::Quaternion:
			value = Quaternion::Identity;
			break;
		case BindingType::Color:
			value = Color(0, 0, 0, 0);
			break;
		case BindingType::ObjectPtr:
			value = ObjectPtr<Object>();
			break;
		}
	}

	void VariantHelper::ReadValue(const BindingType& type, void* ptr, Variant& value)
	{
		switch (type)
		{
		case BindingType::Bool:
			value = *static_cast<bool*>(ptr);
			break;
		case BindingType::Int:
			value = *static_cast<int*>(ptr);
			break;
		case BindingType::Uint:
			value = *static_cast<uint32_t*>(ptr);
			break;
		case BindingType::Float:
			value = *static_cast<float*>(ptr);
			break;
		case BindingType::Enum:
			value = *static_cast<int*>(ptr);
			break;
		case BindingType::String:
			value = *static_cast<String*>(ptr);
			break;
		case BindingType::Vector2:
			value = *static_cast<Vector2*>(ptr);
			break;
		case BindingType::Vector3:
			value = *static_cast<Vector3*>(ptr);
			break;
		case BindingType::Vector4:
			value = *static_cast<Vector4*>(ptr);
			break;
		case BindingType::Quaternion:
			value = *static_cast<Quaternion*>(ptr);
			break;
		case BindingType::Color:
			value = *static_cast<Color*>(ptr);
			break;
		case BindingType::ObjectPtr:
			value = *static_cast<ObjectPtr<Object>*>(ptr);
			break;
		}
	}

	void VariantHelper::WriteValue(const BindingType& type, void* ptr, Variant& value)
	{
		switch (type)
		{
		case BindingType::Bool:
			*static_cast<bool*>(ptr) = std::get<bool>(value);
			break;
		case BindingType::Int:
			*static_cast<int*>(ptr) = std::get<int>(value);
			break;
		case BindingType::Uint:
			*static_cast<uint32_t*>(ptr) = std::get<uint32_t>(value);
			break;
		case BindingType::Float:
			*static_cast<float*>(ptr) = std::get<float>(value);
			break;
		case BindingType::Enum:
			*static_cast<int*>(ptr) = std::get<int>(value);
			break;
		case BindingType::String:
			*static_cast<String*>(ptr) = std::get<String>(value);
			break;
		case BindingType::Vector2:
			*static_cast<Vector2*>(ptr) = std::get<Vector2>(value);
			break;
		case BindingType::Vector3:
			*static_cast<Vector3*>(ptr) = std::get<Vector3>(value);
			break;
		case BindingType::Vector4:
			*static_cast<Vector4*>(ptr) = std::get<Vector4>(value);
			break;
		case BindingType::Quaternion:
			*static_cast<Quaternion*>(ptr) = std::get<Quaternion>(value);
			break;
		case BindingType::Color:
			*static_cast<Color*>(ptr) = std::get<Color>(value);
			break;
		case BindingType::ObjectPtr:
			*static_cast<ObjectPtr<Object>*>(ptr) = std::get<ObjectPtr<Object>>(value);
			break;
		}
	}
}