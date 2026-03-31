#pragma once

#include "Blueberry\Core\Base.h"

namespace Blueberry
{
	struct WindowProperties
	{
		WindowProperties(const String& title, int width, int height, void* data, bool canDropFiles = false) : title(title), width(width), height(height), data(data), canDropFiles(canDropFiles)
		{
		}

		String title;
		int width;
		int height;
		void* data;
		bool canDropFiles;
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
		void SetScreenSize(uint32_t width, uint32_t height, float scale);
	};
}