#pragma once

namespace Blueberry
{
	class BaseCamera;
	class GfxConstantBuffer;

	class PerDrawConstantBuffer
	{
	public:
		static void BindData(BaseCamera* camera);

	private:
		static inline GfxConstantBuffer* s_ConstantBuffer = nullptr;
	};
}