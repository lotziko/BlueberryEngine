#pragma once

#include "Blueberry\Core\ObjectPtr.h"

namespace Blueberry
{
	class Scene;
	class BaseCamera;
	class Texture2D;
	class Material;
	class Transform;
	class Component;
	class ObjectInspector;

	class IconRenderer
	{
		struct IconInfo
		{
			ObjectPtr<Transform> transform;
			ObjectPtr<Component> component;
			ObjectInspector* inspector;
		};

	public:
		static bool Initialize();
		static void Shutdown();
		static void Draw(Scene* scene, BaseCamera* camera);

	private:
		static void ClearCache();
		static void GenerateCache(Scene* scene);

	private:
		static inline Material* s_IconMaterial = nullptr;
		static std::vector<IconInfo> s_IconsCache;
	};
}