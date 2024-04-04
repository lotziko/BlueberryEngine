#include "bbpch.h"
#include "YamlSerializers.h"

namespace DirectX::SimpleMath
{
	void write(ryml::NodeRef* n, const Vector2& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("x") << val.x;
		n->append_child() << ryml::key("y") << val.y;
	}

	bool read(const ryml::ConstNodeRef& n, Vector2* val)
	{
		n["x"] >> val->x;
		n["y"] >> val->y;
		return true;
	}

	void write(ryml::NodeRef* n, const Vector3& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("x") << val.x;
		n->append_child() << ryml::key("y") << val.y;
		n->append_child() << ryml::key("z") << val.z;
	}

	bool read(const ryml::ConstNodeRef& n, Vector3* val)
	{
		n["x"] >> val->x;
		n["y"] >> val->y;
		n["z"] >> val->z;
		return true;
	}

	void write(ryml::NodeRef* n, const Vector4& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("x") << val.x;
		n->append_child() << ryml::key("y") << val.y;
		n->append_child() << ryml::key("z") << val.z;
		n->append_child() << ryml::key("w") << val.w;
	}

	bool read(const ryml::ConstNodeRef& n, Vector4* val)
	{
		n["x"] >> val->x;
		n["y"] >> val->y;
		n["z"] >> val->z;
		n["w"] >> val->w;
		return true;
	}

	void write(ryml::NodeRef* n, const Quaternion& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("x") << val.x;
		n->append_child() << ryml::key("y") << val.y;
		n->append_child() << ryml::key("z") << val.z;
		n->append_child() << ryml::key("w") << val.w;
	}

	bool read(const ryml::ConstNodeRef& n, Quaternion* val)
	{
		n["x"] >> val->x;
		n["y"] >> val->y;
		n["z"] >> val->z;
		n["w"] >> val->w;
		return true;
	}

	void write(ryml::NodeRef* n, const Color& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("r") << val.x;
		n->append_child() << ryml::key("g") << val.y;
		n->append_child() << ryml::key("b") << val.z;
		n->append_child() << ryml::key("a") << val.w;
	}

	bool read(const ryml::ConstNodeRef& n, Color* val)
	{
		n["r"] >> val->x;
		n["g"] >> val->y;
		n["b"] >> val->z;
		n["a"] >> val->w;
		return true;
	}

	bool from_chars(ryml::csubstr buf, Color *c)
	{
		size_t ret = ryml::unformat(buf, "{{}, {}, {}, {}}", c->x, c->y, c->z, c->w); return ret != ryml::yml::npos;
	}
}

namespace Blueberry
{
	size_t to_chars(ryml::substr buf, Guid val)
	{
		char dst[33];
		ByteConverter::BytesToHexString(val.data, dst, sizeof(val.data));
		dst[32] = '\0';
		return ryml::format(buf, dst);
	}

	bool from_chars(ryml::csubstr buf, Guid* v)
	{
		char dst[32];
		memcpy(dst, buf.str, 32);
		ByteConverter::HexStringToBytes(dst, v->data, sizeof(dst));
		return true;
	}

	size_t to_chars(ryml::substr buf, Blueberry::ByteData val)
	{
		char* dst = new char[val.size * 2];
		ByteConverter::BytesToHexString(val.data, dst, val.size);
		return ryml::format(buf, ryml::substr(dst, val.size * 2));
	}

	bool from_chars(ryml::csubstr buf, Blueberry::ByteData* v)
	{
		size_t size = buf.size();
		v->data = new byte[size / 2];
		v->size = size / (2 * sizeof(byte));
		ByteConverter::HexStringToBytes(buf.data(), v->data, size);
		return true;
	}

	void write(ryml::NodeRef* n, const ObjectPtrData& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("fileId") << val.fileId;
		if (val.isAsset)
		{
			n->append_child() << ryml::key("guid") << val.guid;
		}
	}

	bool read(const ryml::ConstNodeRef& n, ObjectPtrData* val)
	{
		n["fileId"] >> val->fileId;
		if (n.has_child("guid"))
		{
			val->isAsset = true;
			n["guid"] >> val->guid;
		}
		else
		{
			val->isAsset = false;
		}
		return true;
	}
}
