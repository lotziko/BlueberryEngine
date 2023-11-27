#include "bbpch.h"
#include "YamlSerializers.h"

namespace DirectX::SimpleMath
{
	size_t to_chars(ryml::substr buf, Vector2 v)
	{
		return ryml::format(buf, "{{}, {}}", v.x, v.y);
	}

	bool from_chars(ryml::csubstr buf, Vector2* v)
	{
		size_t ret = ryml::unformat(buf, "{{}, {}}", v->x, v->y); return ret != ryml::yml::npos;
	}

	size_t to_chars(ryml::substr buf, Vector3 v)
	{
		return ryml::format(buf, "{{}, {}, {}}", v.x, v.y, v.z);
	}

	bool from_chars(ryml::csubstr buf, Vector3* v)
	{
		size_t ret = ryml::unformat(buf, "{{}, {}, {}}", v->x, v->y, v->z); return ret != ryml::yml::npos;
	}

	size_t to_chars(ryml::substr buf, Vector4 v)
	{
		return ryml::format(buf, "{{}, {}, {}, {}}", v.x, v.y, v.z, v.w);
	}

	bool from_chars(ryml::csubstr buf, Vector4* v)
	{
		size_t ret = ryml::unformat(buf, "{{}, {}, {}, {}}", v->x, v->y, v->z, v->w); return ret != ryml::yml::npos;
	}

	size_t to_chars(ryml::substr buf, Quaternion q)
	{
		return ryml::format(buf, "{{}, {}, {}, {}}", q.x, q.y, q.z, q.w);
	}

	bool from_chars(ryml::csubstr buf, Quaternion* q)
	{
		size_t ret = ryml::unformat(buf, "{{}, {}, {}, {}}", q->x, q->y, q->z, q->w); return ret != ryml::yml::npos;
	}
}

namespace Blueberry
{
	size_t to_chars(ryml::substr buf, Guid val)
	{
		char dst[32];
		ByteConverter::BytesToHexString(val.data, dst, sizeof(val.data));
		return ryml::format(buf, dst);
	}

	bool from_chars(ryml::csubstr buf, Guid* v)
	{
		char dst[32];
		memcpy(dst, buf.str, 32);
		ByteConverter::HexStringToBytes(dst, v->data, sizeof(dst));
		return true;
	}
}
