#include "OpenXRRendererDX11.h"

// Tell OpenXR what platform code we'll be using
#define XR_USE_PLATFORM_WIN32
#define XR_USE_GRAPHICS_API_D3D11

#include "Blueberry\Core\Engine.h"
#include "Blueberry\Graphics\GfxDevice.h"
#include "..\..\Blueberry\Graphics\RenderContext.h"
#include "..\DX11\GfxDeviceDX11.h"
#include "..\DX11\GfxTextureDX11.h"

#include "openxr\openxr.h"
#include "openxr\openxr_platform.h"

#include "Blueberry\Scene\Components\Camera.h"

namespace Blueberry
{
	struct swapchain_surfdata_t 
	{
		ID3D11Resource *texture;
		ID3D11DepthStencilView *depthView;
		ID3D11RenderTargetView *targetView;
	};

	struct swapchain_t
	{
		XrSwapchain handle;
		int32_t width;
		int32_t height;
		List<XrSwapchainImageD3D11KHR> surfaceImages;
		List<swapchain_surfdata_t> surfaceData;
	};

	struct CompositionData
	{
		bool hasLayer;
		XrCompositionLayerProjection layerProjection;
		List<XrCompositionLayerProjectionView> layerProjectionViews;
		uint32_t imgId;
	} s_CompositionData;

	static Engine* s_Engine;
	static ID3D11Device* s_Device;
	static ID3D11DeviceContext* s_DeviceContext;

	static PFN_xrGetD3D11GraphicsRequirementsKHR s_ExtXrGetD3D11GraphicsRequirementsKHR = nullptr;
	static PFN_xrCreateDebugUtilsMessengerEXT s_ExtXrCreateDebugUtilsMessengerEXT = nullptr;
	static PFN_xrDestroyDebugUtilsMessengerEXT s_ExtXrDestroyDebugUtilsMessengerEXT = nullptr;

	static XrFormFactor s_AppConfigForm = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	static XrViewConfigurationType s_AppConfigView = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;

	const XrPosef XR_POSE_IDENTITY = { {0,0,0,1}, {0,0,0} };
	const int64_t SWAPCHAIN_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	static XrInstance s_XrInstance = {};
	static XrSession s_XrSession = {};
	static XrSessionState s_XrSessionState = XR_SESSION_STATE_UNKNOWN;
	static bool s_XrRunning = false;
	static XrSpace s_XrAppSpace = {};
	static XrSystemId s_XrSystemId = XR_NULL_SYSTEM_ID;
	static XrEnvironmentBlendMode s_XrBlend = {};
	static XrDebugUtilsMessengerEXT s_XrDebug = {};

	static List<XrView> s_XrViews;
	static List<XrViewConfigurationView> s_XrConfigViews;
	static List<swapchain_t> s_XrSwapchains;

	static XrFrameState s_XrFrameState;

	static void WaitForFrame()
	{
		bool exit = false;
		XrEventDataBuffer event_buffer = { XR_TYPE_EVENT_DATA_BUFFER };

		while (xrPollEvent(s_XrInstance, &event_buffer) == XR_SUCCESS) {
			switch (event_buffer.type) {
			case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED: {
				XrEventDataSessionStateChanged *changed = (XrEventDataSessionStateChanged*)&event_buffer;
				s_XrSessionState = changed->state;

				// Session state change is where we can begin and end sessions, as well as find quit messages!
				switch (s_XrSessionState) {
				case XR_SESSION_STATE_READY: {
					XrSessionBeginInfo begin_info = { XR_TYPE_SESSION_BEGIN_INFO };
					begin_info.primaryViewConfigurationType = s_AppConfigView;
					xrBeginSession(s_XrSession, &begin_info);
					s_XrRunning = true;
				} break;
				case XR_SESSION_STATE_STOPPING: {
					s_XrRunning = false;
					xrEndSession(s_XrSession);
				} break;
				case XR_SESSION_STATE_EXITING:      exit = true;              break;
				case XR_SESSION_STATE_LOSS_PENDING: exit = true;              break;
				}
			} break;
			case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING: exit = true; return;
			}
			event_buffer = { XR_TYPE_EVENT_DATA_BUFFER };
		}

		s_XrFrameState = { XR_TYPE_FRAME_STATE };
		xrWaitFrame(s_XrSession, nullptr, &s_XrFrameState);
	}

	swapchain_surfdata_t CreateSurfaceData(XrBaseInStructure& swapchainImg)
	{
		swapchain_surfdata_t result = {};

		// Get information about the swapchain image that OpenXR made for us!
		XrSwapchainImageD3D11KHR& swapchainImgDX11 = reinterpret_cast<XrSwapchainImageD3D11KHR&>(swapchainImg);
		D3D11_TEXTURE2D_DESC color_desc;
		swapchainImgDX11.texture->GetDesc(&color_desc);
		result.texture = swapchainImgDX11.texture;

		// Create a view resource for the swapchain image target that we can use to set up rendering.
		D3D11_RENDER_TARGET_VIEW_DESC targetDesc = {};
		targetDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
		targetDesc.Texture2DArray.ArraySize = 2;

		targetDesc.Format = static_cast<DXGI_FORMAT>(SWAPCHAIN_FORMAT);
		s_Device->CreateRenderTargetView(swapchainImgDX11.texture, &targetDesc, &result.targetView);

		return result;
	}

	bool OpenXRRendererDX11::InitializeImpl()
	{
		GfxDeviceDX11* dxDevice = static_cast<GfxDeviceDX11*>(GfxDevice::GetInstance());
		s_Engine = Engine::GetInstance();
		s_Device = dxDevice->GetDevice();
		s_DeviceContext = dxDevice->GetDeviceContext();

		List<const char*> useExtensions;
		const char* askExtensions[] = 
		{
			XR_KHR_D3D11_ENABLE_EXTENSION_NAME, // Use Direct3D11 for rendering
			XR_EXT_DEBUG_UTILS_EXTENSION_NAME,  // Debug utils for extra info
		};

		uint32_t extCount = 0;
		xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr);
		List<XrExtensionProperties> xrExts(extCount, { XR_TYPE_EXTENSION_PROPERTIES });
		xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, xrExts.data());

		for (size_t i = 0; i < xrExts.size(); i++)
		{
			// Check if we're asking for this extensions, and add it to our use 
			// list!
			for (int32_t ask = 0; ask < _countof(askExtensions); ask++)
			{
				if (strcmp(askExtensions[ask], xrExts[i].extensionName) == 0)
				{
					useExtensions.push_back(askExtensions[ask]);
					break;
				}
			}
		}

		// If a required extension isn't present, you want to ditch out here!
		// It's possible something like your rendering API might not be provided
		// by the active runtime. APIs like OpenGL don't have universal support.
		if (!std::any_of(useExtensions.begin(), useExtensions.end(), [](const char *ext) {
			return strcmp(ext, XR_KHR_D3D11_ENABLE_EXTENSION_NAME) == 0;
		}))
		{
			return false;
		}

		// Initialize OpenXR with the extensions we've found!
		XrInstanceCreateInfo createInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
		createInfo.enabledExtensionCount = static_cast<uint32_t>(useExtensions.size());
		createInfo.enabledExtensionNames = useExtensions.data();
		createInfo.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
		strcpy_s(createInfo.applicationInfo.applicationName, "Test App");
		xrCreateInstance(&createInfo, &s_XrInstance);

		// Check if OpenXR is on this system, if this is null here, the user 
		// needs to install an OpenXR runtime and ensure it's active!
		if (s_XrInstance == XR_NULL_HANDLE)
		{
			return false;
		}

		xrGetInstanceProcAddr(s_XrInstance, "xrCreateDebugUtilsMessengerEXT", (PFN_xrVoidFunction *)(&s_ExtXrCreateDebugUtilsMessengerEXT));
		xrGetInstanceProcAddr(s_XrInstance, "xrDestroyDebugUtilsMessengerEXT", (PFN_xrVoidFunction *)(&s_ExtXrDestroyDebugUtilsMessengerEXT));
		xrGetInstanceProcAddr(s_XrInstance, "xrGetD3D11GraphicsRequirementsKHR", (PFN_xrVoidFunction *)(&s_ExtXrGetD3D11GraphicsRequirementsKHR));

		XrDebugUtilsMessengerCreateInfoEXT debugInfo = { XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
		debugInfo.messageTypes =
			XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;
		debugInfo.messageSeverities =
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugInfo.userCallback = [](XrDebugUtilsMessageSeverityFlagsEXT severity, XrDebugUtilsMessageTypeFlagsEXT types, const XrDebugUtilsMessengerCallbackDataEXT *msg, void* user_data) {
			// Print the debug message we got! There's a bunch more info we could
			// add here too, but this is a pretty good start, and you can always
			// add a breakpoint this line!
			printf("%s: %s\n", msg->functionName, msg->message);

			// Output to debug window
			char text[512];
			sprintf_s(text, "%s: %s", msg->functionName, msg->message);
			OutputDebugStringA(text);

			// Returning XR_TRUE here will force the calling function to fail
			return static_cast<XrBool32>(XR_FALSE);
		};

		if (s_ExtXrCreateDebugUtilsMessengerEXT)
		{
			s_ExtXrCreateDebugUtilsMessengerEXT(s_XrInstance, &debugInfo, &s_XrDebug);
		}

		// Request a form factor from the device (HMD, Handheld, etc.)
		XrSystemGetInfo systemInfo = { XR_TYPE_SYSTEM_GET_INFO };
		systemInfo.formFactor = s_AppConfigForm;
		xrGetSystem(s_XrInstance, &systemInfo, &s_XrSystemId);

		// Check what blend mode is valid for this device (opaque vs transparent displays)
		// We'll just take the first one available!
		uint32_t blendCount = 0;
		xrEnumerateEnvironmentBlendModes(s_XrInstance, s_XrSystemId, s_AppConfigView, 1, &blendCount, &s_XrBlend);

		// OpenXR wants to ensure apps are using the correct graphics card, so this MUST be called 
		// before xrCreateSession. This is crucial on devices that have multiple graphics cards, 
		// like laptops with integrated graphics chips in addition to dedicated graphics cards.
		XrGraphicsRequirementsD3D11KHR requirement = { XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
		s_ExtXrGetD3D11GraphicsRequirementsKHR(s_XrInstance, s_XrSystemId, &requirement);
		
		// A session represents this application's desire to display things! This is where we hook up our graphics API.
		// This does not start the session, for that, you'll need a call to xrBeginSession, which we do in openxr_poll_events
		XrGraphicsBindingD3D11KHR binding = { XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };
		binding.device = s_Device;
		XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
		sessionInfo.next = &binding;
		sessionInfo.systemId = s_XrSystemId;
		xrCreateSession(s_XrInstance, &sessionInfo, &s_XrSession);

		// Unable to start a session, may not have an MR device attached or ready
		if (s_XrSession == XR_NULL_HANDLE)
		{
			return false;
		}
		s_Engine->AddWaitFrameCallback(&WaitForFrame);

		// OpenXR uses a couple different types of reference frames for positioning content, we need to choose one for
		// displaying our content! STAGE would be relative to the center of your guardian system's bounds, and LOCAL
		// would be relative to your device's starting location. HoloLens doesn't have a STAGE, so we'll use LOCAL.
		XrReferenceSpaceCreateInfo refSpace = { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
		refSpace.poseInReferenceSpace = XR_POSE_IDENTITY;
		refSpace.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
		xrCreateReferenceSpace(s_XrSession, &refSpace, &s_XrAppSpace);

		// Now we need to find all the viewpoints we need to take care of! For a stereo headset, this should be 2.
		// Similarly, for an AR phone, we'll need 1, and a VR cave could have 6, or even 12!
		uint32_t viewCount = 0;
		xrEnumerateViewConfigurationViews(s_XrInstance, s_XrSystemId, s_AppConfigView, 0, &viewCount, nullptr);
		s_XrConfigViews.resize(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
		s_XrViews.resize(viewCount, { XR_TYPE_VIEW });
		xrEnumerateViewConfigurationViews(s_XrInstance, s_XrSystemId, s_AppConfigView, viewCount, &viewCount, s_XrConfigViews.data());
		
		const XrViewConfigurationView& view = s_XrConfigViews[0];
		
		XrSwapchainCreateInfo swapchainInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
		XrSwapchain handle;
		swapchainInfo.arraySize = 2;
		swapchainInfo.mipCount = 1;
		swapchainInfo.faceCount = 1;
		swapchainInfo.format = SWAPCHAIN_FORMAT;
		swapchainInfo.width = view.recommendedImageRectWidth;
		swapchainInfo.height = view.recommendedImageRectHeight;
		swapchainInfo.sampleCount = view.recommendedSwapchainSampleCount;
		swapchainInfo.usageFlags = XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
		xrCreateSwapchain(s_XrSession, &swapchainInfo, &handle);

		// Find out how many textures were generated for the swapchain
		uint32_t surface—ount = 0;
		xrEnumerateSwapchainImages(handle, 0, &surface—ount, nullptr);

		swapchain_t swapchain = {};
		swapchain.width = swapchainInfo.width;
		swapchain.height = swapchainInfo.height;
		swapchain.handle = handle;
		swapchain.surfaceImages.resize(surface—ount, { XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR });
		swapchain.surfaceData.resize(surface—ount);
		xrEnumerateSwapchainImages(swapchain.handle, surface—ount, &surface—ount, reinterpret_cast<XrSwapchainImageBaseHeader*>(swapchain.surfaceImages.data()));
		for (uint32_t i = 0; i < surface—ount; i++)
		{
			swapchain.surfaceData[i] = CreateSurfaceData(reinterpret_cast<XrBaseInStructure&>(swapchain.surfaceImages[i]));
		}
		s_XrSwapchains.push_back(swapchain);

		return true;
	}

	void OpenXRRendererDX11::ShutdownImpl()
	{
		if (s_XrAppSpace != XR_NULL_HANDLE)
		{
			xrDestroySpace(s_XrAppSpace);
			s_XrAppSpace = XR_NULL_HANDLE;
		}
		if (s_XrSession != XR_NULL_HANDLE)
		{
			xrDestroySession(s_XrSession);
			s_XrSession = XR_NULL_HANDLE;
		}
		if (s_XrDebug != XR_NULL_HANDLE)
		{
			s_ExtXrDestroyDebugUtilsMessengerEXT(s_XrDebug);
			s_XrDebug = XR_NULL_HANDLE;
		}
		if (s_XrInstance != XR_NULL_HANDLE)
		{
			xrDestroyInstance(s_XrInstance);
			s_XrInstance = XR_NULL_HANDLE;
		}
		for (swapchain_t& swapchain : s_XrSwapchains)
		{
			for (swapchain_surfdata_t& surfData : swapchain.surfaceData)
			{
				surfData.targetView->Release();
			}
			swapchain.surfaceData.clear();
			xrDestroySwapchain(swapchain.handle);
		}
		s_XrSwapchains.clear();
		s_Engine->RemoveWaitFrameCallback(&WaitForFrame);
		s_XrRunning = false;
		s_XrSessionState = XR_SESSION_STATE_UNKNOWN;
		s_XrSystemId = XR_NULL_SYSTEM_ID;
	}

	bool OpenXRRendererDX11::IsActiveImpl()
	{
		return s_XrSession != XR_NULL_HANDLE && s_XrRunning && (s_XrSessionState == XR_SESSION_STATE_VISIBLE || s_XrSessionState == XR_SESSION_STATE_FOCUSED);
	}

	Matrix GetProjectionMatrix(XrFovf fov, float nearPlane, float farPlane)
	{
		const float left = nearPlane * tanf(fov.angleLeft);
		const float right = nearPlane * tanf(fov.angleRight);
		const float down = nearPlane * tanf(fov.angleDown);
		const float up = nearPlane * tanf(fov.angleUp);
		return Matrix::CreatePerspectiveOffCenter(left, right, down, up, nearPlane, farPlane);
	}

	void OpenXRRendererDX11::BeginFrameImpl()
	{
		if (s_XrSession == XR_NULL_HANDLE)
		{
			return;
		}
		
		xrBeginFrame(s_XrSession, nullptr);

		XrCompositionLayerProjection& layerProj = s_CompositionData.layerProjection = { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
		List<XrCompositionLayerProjectionView>& views = s_CompositionData.layerProjectionViews;
		
		s_CompositionData.hasLayer = false;
		s_CompositionData.imgId = 0;
		bool sessionActive = s_XrSessionState == XR_SESSION_STATE_VISIBLE || s_XrSessionState == XR_SESSION_STATE_FOCUSED;
		if (!sessionActive)
		{
			return;
		}

		// Find the state and location of each viewpoint at the predicted time
		uint32_t viewCount = 0;
		XrViewState viewState = { XR_TYPE_VIEW_STATE };
		XrViewLocateInfo locateInfo = { XR_TYPE_VIEW_LOCATE_INFO };
		locateInfo.viewConfigurationType = s_AppConfigView;
		locateInfo.displayTime = s_XrFrameState.predictedDisplayTime;
		locateInfo.space = s_XrAppSpace;
		xrLocateViews(s_XrSession, &locateInfo, &viewState, static_cast<uint32_t>(s_XrViews.size()), &viewCount, s_XrViews.data());
		views.resize(viewCount);

		// We need to ask which swapchain image to use for rendering! Which one will we get?
		// Who knows! It's up to the runtime to decide.
		uint32_t imgId;
		XrSwapchainImageAcquireInfo acquireInfo = { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
		xrAcquireSwapchainImage(s_XrSwapchains[0].handle, &acquireInfo, &imgId);

		// Wait until the image is available to render to. The compositor could still be
		// reading from it.
		XrSwapchainImageWaitInfo waitInfo = { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
		waitInfo.timeout = XR_INFINITE_DURATION;
		xrWaitSwapchainImage(s_XrSwapchains[0].handle, &waitInfo);

		s_CompositionData.imgId = imgId;

		for (uint32_t i = 0; i < viewCount; i++)
		{
			// Set up our rendering information for the viewpoint we're using right now!
			views[i] = { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
			views[i].pose = s_XrViews[i].pose;
			views[i].fov = s_XrViews[i].fov;
			views[i].subImage.imageArrayIndex = i;
			views[i].subImage.swapchain = s_XrSwapchains[0].handle;
			views[i].subImage.imageRect.offset = { 0, 0 };
			views[i].subImage.imageRect.extent = { s_XrSwapchains[0].width, s_XrSwapchains[0].height };

			auto orientation = views[i].pose.orientation;
			auto position = views[i].pose.position;

			DirectX::XMVECTOR rotation = DirectX::XMVectorSet(-orientation.x, -orientation.y, orientation.z, orientation.w);
			DirectX::XMVECTOR translation = DirectX::XMVectorSet(position.x,position.y, -position.z, 1.0f);

			m_MultiviewViewMatrix[i] = DirectX::XMMatrixInverse(nullptr, DirectX::XMMatrixAffineTransformation(DirectX::g_XMOne, DirectX::g_XMZero, rotation, translation));//(Matrix::CreateFromQuaternion(Quaternion(orientation.x, orientation.y, orientation.z, orientation.w)) * Matrix::CreateTranslation(position.x, position.y, position.z)).Invert();
			m_MultiviewProjectionMatrix[i] = GetProjectionMatrix(s_XrViews[i].fov, 0.05f, 100.0f);
		}
		m_MultiviewViewport = { 0, 0, s_XrSwapchains[0].width, s_XrSwapchains[0].height };

		layerProj.space = s_XrAppSpace;
		layerProj.viewCount = static_cast<uint32_t>(views.size());
		layerProj.views = views.data();
		s_CompositionData.hasLayer = true;

		m_SubmittedColorRenderTarget = nullptr;
	}

	void OpenXRRendererDX11::FillCameraDataImpl(CameraData& cameraData)
	{
		if (s_XrSession == XR_NULL_HANDLE)
		{
			return;
		}

		cameraData.size = Vector2Int(m_MultiviewViewport.width, m_MultiviewViewport.height);
		cameraData.isMultiview = true;
		for (uint32_t i = 0; i < 2; i++)
		{
			cameraData.multiviewViewMatrix[i] = m_MultiviewViewMatrix[i];
			cameraData.multiviewProjectionMatrix[i] = m_MultiviewProjectionMatrix[i];
		}
		cameraData.multiviewViewport = m_MultiviewViewport;
	}

	void OpenXRRendererDX11::SubmitColorRenderTargetImpl(GfxTexture* renderTarget)
	{
		if (s_XrSession == XR_NULL_HANDLE)
		{
			return;
		}

		m_SubmittedColorRenderTarget = renderTarget;
		uint32_t imgId = s_CompositionData.imgId;

		ID3D11Resource* source = (static_cast<GfxTextureDX11*>(renderTarget))->GetTexture();
		ID3D11Resource* target = s_XrSwapchains[0].surfaceData[imgId].texture;
		if (target != nullptr)
		{
			s_DeviceContext->CopyResource(target, source);
		}
	}

	void OpenXRRendererDX11::EndFrameImpl()
	{
		if (s_XrSession == XR_NULL_HANDLE)
		{
			return;
		}

		uint32_t imgId = s_CompositionData.imgId;

		if (m_SubmittedColorRenderTarget == nullptr)
		{
			s_DeviceContext->ClearRenderTargetView(s_XrSwapchains[0].surfaceData[imgId].targetView, Color(0, 0, 0, 0));
		}

		// And tell OpenXR we're done with rendering to this one!
		for (uint32_t i = 0; i < s_XrSwapchains.size(); i++)
		{
			XrSwapchainImageReleaseInfo releaseInfo = { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
			xrReleaseSwapchainImage(s_XrSwapchains[i].handle, &releaseInfo);
		}

		// We're finished with rendering our layer, so send it off for display!
		XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
		endInfo.displayTime = s_XrFrameState.predictedDisplayTime;
		endInfo.environmentBlendMode = s_XrBlend;
		if (s_CompositionData.hasLayer)
		{
			XrCompositionLayerBaseHeader* layer = (XrCompositionLayerBaseHeader*)&s_CompositionData.layerProjection;
			endInfo.layerCount = 1;
			endInfo.layers = &layer;
		}
		else
		{
			endInfo.layerCount = 0;
			endInfo.layers = nullptr;
		}
		xrEndFrame(s_XrSession, &endInfo);
	}
}
