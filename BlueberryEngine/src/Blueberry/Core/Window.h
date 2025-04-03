#pragma once

namespace Blueberry
{
	struct WindowProperties
	{
		std::string Title;
		int Width;
		int Height;
		void* Data;

		WindowProperties(std::string title, int width, int height, void* data) : Title(title), Width(width), Height(height), Data(data)
		{
		}
	};

	class Window
	{
	public:
		virtual ~Window() = default;

		virtual bool IsActive() = 0;
		virtual bool ProcessMessages() = 0;

		virtual void* GetHandle() = 0;
		
		virtual int GetWidth() const = 0;
		virtual int GetHeight() const = 0;

		static Window* Create(const WindowProperties& properties);

	protected:
		void SetScreenSize(const uint32_t& width, const uint32_t& height, const float& scale);
	};
}