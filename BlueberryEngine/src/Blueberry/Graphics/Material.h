#pragma once

namespace Blueberry
{
	class Shader;

	class Material : public Object
	{
	private:
		Shader* m_Shader;
	};
}