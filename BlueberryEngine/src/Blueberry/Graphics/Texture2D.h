#pragma once
#include "Blueberry\Graphics\Structs.h"
#include "Blueberry\Graphics\Texture.h"

namespace Blueberry
{
	class Texture2D : public Texture
	{
		OBJECT_DECLARATION(Texture2D)

	public:
		Texture2D() = default;
		Texture2D(const TextureProperties& properties);

		virtual void Serialize(SerializationContext& context, ryml::NodeRef& node) override final;
		virtual void Deserialize(SerializationContext& context, ryml::NodeRef& node) override final;

		static Ref<Texture2D> Create(const TextureProperties& properties);
	};
}