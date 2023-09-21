#include "bbpch.h"
#include "RegisterGraphicsTypes.h"

#include "Blueberry\Core\ClassDB.h"

#include "Blueberry\Graphics\Texture.h"
#include "Blueberry\Graphics\Texture2D.h"
#include "Blueberry\Graphics\Mesh.h"
#include "Blueberry\Graphics\Shader.h"
#include "Blueberry\Graphics\Material.h"

namespace Blueberry
{
	void RegisterGraphicsTypes()
	{
		REGISTER_ABSTRACT_CLASS(Texture);
		REGISTER_CLASS(Texture2D);
		REGISTER_CLASS(Mesh);
		REGISTER_CLASS(Shader);
		REGISTER_CLASS(Material);
	}
}