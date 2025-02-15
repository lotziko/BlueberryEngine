#pragma once

namespace Blueberry
{
	class ImGuiRenderer
	{
	public:
		static bool Initialize();
		static void Shutdown();

		static void Begin();
		static void End();

	protected:
		virtual bool InitializeImpl() = 0;
		virtual void ShutdownImpl() = 0;

		virtual void BeginImpl() = 0;
		virtual void EndImpl() = 0;

	private:
		static inline ImGuiRenderer* s_Instance = nullptr;
	};
}