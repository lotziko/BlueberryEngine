#pragma once

#include "Blueberry\Math\Math.h"
#include "Blueberry\Core\ObjectPtr.h"
#include "Blueberry\Core\Structs.h"
#include "Blueberry\Core\Guid.h"
#include "Blueberry\Tools\ByteConverter.h"

#include <rapidyaml\ryml.h>

namespace DirectX::SimpleMath
{
	void write(ryml::NodeRef* n, const Vector2& val);
	bool read(const ryml::ConstNodeRef& n, Vector2* val);

	void write(ryml::NodeRef* n, const Vector3& val);
	bool read(const ryml::ConstNodeRef& n, Vector3* val);

	void write(ryml::NodeRef* n, const Vector4& val);
	bool read(const ryml::ConstNodeRef& n, Vector4* val);

	void write(ryml::NodeRef* n, const Quaternion& val);
	bool read(const ryml::ConstNodeRef& n, Quaternion* val);

	void write(ryml::NodeRef* n, const Color& val);
	bool read(const ryml::ConstNodeRef& n, Color* val);
}

namespace DirectX
{
	void write(ryml::NodeRef* n, const BoundingBox& val);
	bool read(const ryml::ConstNodeRef& n, BoundingBox* val);
}

namespace Blueberry
{
	size_t to_chars(ryml::substr buf, Blueberry::Guid val);
	bool from_chars(ryml::csubstr buf, Blueberry::Guid* v);

	size_t to_chars(ryml::substr buf, Blueberry::ByteData val);
	bool from_chars(ryml::csubstr buf, Blueberry::ByteData* v);

	void write(ryml::NodeRef* n, const ObjectPtrData& val);
	bool read(const ryml::ConstNodeRef& n, ObjectPtrData* val);

	size_t to_chars(ryml::substr buf, Blueberry::String const& val);
	bool from_chars(ryml::csubstr buf, Blueberry::String* v);
}