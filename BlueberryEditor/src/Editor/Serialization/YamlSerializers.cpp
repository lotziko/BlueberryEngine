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
}

namespace DirectX
{
	void write(ryml::NodeRef* n, const BoundingBox& val)
	{
		*n |= ryml::MAP;
		n->append_child() << ryml::key("m_Center") << static_cast<SimpleMath::Vector3>(val.Center);
		n->append_child() << ryml::key("m_Extents") << static_cast<SimpleMath::Vector3>(val.Extents);
	}

	bool read(const ryml::ConstNodeRef& n, BoundingBox* val)
	{
		n["m_Center"].operator>><SimpleMath::Vector3>(static_cast<SimpleMath::Vector3>(val->Center));
		n["m_Extents"].operator>><SimpleMath::Vector3>(static_cast<SimpleMath::Vector3>(val->Extents));
		return true;
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

	size_t to_chars(ryml::substr buf, Blueberry::DataWrapper<ByteData> val)
	{
		uint8_t* data = val.reference.data();
		size_t size = val.reference.size();
		char* dst = BB_MALLOC_ARRAY(char, size * 2);
		ByteConverter::BytesToHexString(data, dst, size);
		size_t result = ryml::format(buf, ryml::substr(dst, size * 2));
		BB_FREE(dst);
		return result;
	}

	bool from_chars(ryml::csubstr buf, Blueberry::DataWrapper<ByteData>* v)
	{
		size_t size = buf.size();
		v->reference.resize(size / (2 * sizeof(uint8_t)));
		ByteConverter::HexStringToBytes(buf.data(), v->reference.data(), size);
		return true;
	}

	size_t to_chars(ryml::substr buf, Blueberry::DataWrapper<Blueberry::List<int>> val)
	{
		int* data = val.reference.data();
		size_t size = val.reference.size() * sizeof(int);
		char* dst = BB_MALLOC_ARRAY(char, size * 2);
		ByteConverter::BytesToHexString(data, dst, size);
		size_t result = ryml::format(buf, ryml::substr(dst, size * 2));
		BB_FREE(dst);
		return result;
	}

	bool from_chars(ryml::csubstr buf, Blueberry::DataWrapper<Blueberry::List<int>>* v)
	{
		size_t size = buf.size();
		v->reference.resize(size / (2 * sizeof(int)));
		ByteConverter::HexStringToBytes(buf.data(), v->reference.data(), size);
		return true;
	}

	size_t to_chars(ryml::substr buf, Blueberry::DataWrapper<Blueberry::List<float>> val)
	{
		float* data = val.reference.data();
		size_t size = val.reference.size() * sizeof(float);
		char* dst = BB_MALLOC_ARRAY(char, size * 2);
		ByteConverter::BytesToHexString(data, dst, size);
		size_t result = ryml::format(buf, ryml::substr(dst, size * 2));
		BB_FREE(dst);
		return result;
	}

	bool from_chars(ryml::csubstr buf, Blueberry::DataWrapper<Blueberry::List<float>>* v)
	{
		size_t size = buf.size();
		v->reference.resize(size / (2 * sizeof(float)));
		ByteConverter::HexStringToBytes(buf.data(), v->reference.data(), size);
		return true;
	}

	void write(ryml::NodeRef* n, const ObjectPtrData& val)
	{
		*n |= ryml::MAP;
		*n |= ryml::_WIP_STYLE_FLOW_SL;
		n->append_child() << ryml::key("fileId") << val.fileId;
		if (val.guid.data[0] > 0)
		{
			n->append_child() << ryml::key("guid") << val.guid;
		}
	}

	bool read(const ryml::ConstNodeRef& n, ObjectPtrData* val)
	{
		n["fileId"] >> val->fileId;
		if (n.has_child("guid"))
		{
			n["guid"] >> val->guid;
		}
		else
		{
			val->guid = {};
		}
		return true;
	}

	// ryml std::string
	bool from_chars(ryml::csubstr buf, Blueberry::String* v)
	{
		v->resize(buf.len);
		if (buf.len)
		{
			memcpy(&(*v)[0], buf.str, buf.len);
		}
		return true;
	}

	// ryml std::string
	size_t to_chars(ryml::substr buf, Blueberry::String const& val)
	{
		size_t len = buf.len < val.size() ? buf.len : val.size();
		if (len)
		{
			memcpy(buf.str, val.data(), len);
		}
		return val.size();
	}
}
