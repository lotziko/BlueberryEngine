#pragma once

#include "Blueberry\Core\Guid.h"
#include "Editor\Serialization\YamlSerializer.h"

namespace Blueberry
{
	class Scene;
	class Entity;

	class YamlSceneSerializer : public YamlSerializer
	{
	public:
		void AddSceneObjects(Scene* scene);

		virtual void Serialize(const String& path) override;
		virtual void Deserialize(const String& path) override;

	private:
		void GatherSceneObjects(Entity* entity);
	};
}