#include "ObjectHelper.h"

#include "VariantHelper.h"

namespace Blueberry
{
	struct PathData
	{
		BindingType bindingType;
		char* context;
		char* fieldTarget;
		const FieldInfo* fieldInfo;
		bool isValid;
	};

	PathData ParsePath(Object* object, const String& path)
	{
		PathData result = {};
		result.context = reinterpret_cast<char*>(object);
		size_t type = object->GetType();

		size_t offset = 0;
		for (size_t i = 0; i < path.size(); ++i)
		{
			if (path[i] == '[')
			{
				String name = path.substr(offset, i - offset);
				const ClassInfo* classInfo = ClassDB::GetInfo(type);
				result.fieldInfo = classInfo->GetField(name);
				if (result.fieldInfo == nullptr)
				{
					return {};
				}
				result.bindingType = VariantHelper::GetChildType(result.fieldInfo->type);
				ListBase* list = result.fieldInfo->Get<ListBase>(result.context);
				i += 1;
				for (size_t j = i; j < path.size(); ++j)
				{
					if (path[j] == ']')
					{
						size_t index = std::stoi(path.substr(i, j - i).c_str());
						result.context = static_cast<char*>(list->get_base(index));
						result.fieldTarget = result.context;
						i = j;
						offset = j + 1;
						break;
					}
				}
			}
			else if (path[i] == '.' || i == path.size() - 1)
			{
				size_t length = (i < path.size() - 1 ? i : path.size()) - offset;
				if (length > 0)
				{
					String name = path.substr(offset, length);
					const ClassInfo* classInfo = ClassDB::GetInfo(type);
					result.fieldInfo = classInfo->GetField(name);
					if (result.fieldInfo == nullptr)
					{
						return {};
					}
					result.bindingType = result.fieldInfo->type;
					result.fieldTarget = result.context + result.fieldInfo->offset;
					offset = i + 1;
				}
			}
		}
		result.isValid = true;
		return result;
	}

	void ObjectHelper::ReadValue(Object* object, const String& path, Variant& value)
	{
		PathData data = ParsePath(object, path);
		if (data.isValid)
		{
			VariantHelper::ReadValue(data.bindingType, data.fieldTarget, value);
		}
	}

	void ObjectHelper::WriteValue(Object* object, const String& path, Variant& value)
	{
		PathData data = ParsePath(object, path);
		if (data.isValid)
		{
			VariantHelper::WriteValue(data.bindingType, data.fieldTarget, value);
			if (data.fieldInfo->options.updateCallback != nullptr)
			{
				data.fieldInfo->options.updateCallback->Invoke(data.context);
			}
		}
	}
}