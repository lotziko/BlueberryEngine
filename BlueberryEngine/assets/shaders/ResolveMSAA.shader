Shader
{
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex ResolveMSAAVertex
		#pragma fragment ResolveMSAAFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		struct Output
		{
			float4 color : SV_TARGET;
			float depth : SV_DEPTH;
		};

		Varyings ResolveMSAAVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		TEXTURE2D_X_MSAA(_ScreenNormalTexture, 4);
		TEXTURE2D_X_MSAA_FLOAT(_ScreenDepthStencilTexture, 4);
		
		Output ResolveMSAAFragment(Varyings input)
		{
			SETUP_INPUT_VIEW_INDEX(input);

			Output output;
			for (int i = 0; i < 4; ++i)
			{
				uint2 uv = uint2(input.texcoord.x * CAMERA_SIZE_INV_SIZE.x, input.texcoord.y * CAMERA_SIZE_INV_SIZE.y);
				output.color += LOAD_TEXTURE2D_X_MSAA(_ScreenNormalTexture, uv, i);
				output.depth += LOAD_TEXTURE2D_X_MSAA(_ScreenDepthStencilTexture, uv, i);
			}
			output.color /= 4;
			output.depth /= 4;
			return output;
		}
		HLSLEND
	}
	Pass
	{
		Blend SrcAlpha OneMinusSrcAlpha
		ZWrite On
		Cull None

		HLSLBEGIN
		#pragma vertex ResolveMSAAVertex
		#pragma fragment ResolveMSAAFragment

		#pragma keyword_global_vertex MULTIVIEW
		#pragma keyword_global_fragment MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		struct Output
		{
			float4 color : SV_TARGET;
		};

		Varyings ResolveMSAAVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			return output;
		}

		TEXTURE2D_X_MSAA(_ScreenColorTexture, 4);

		Output ResolveMSAAFragment(Varyings input)
		{
			SETUP_INPUT_VIEW_INDEX(input);

			Output output;
			for (int i = 0; i < 4; ++i)
			{
				uint2 uv = uint2(input.texcoord.x * CAMERA_SIZE_INV_SIZE.x, input.texcoord.y * CAMERA_SIZE_INV_SIZE.y);
				output.color += LOAD_TEXTURE2D_X_MSAA(_ScreenColorTexture, uv, i);
			}
			output.color /= 4;
			// Gamma correction
			output.color = float4(pow(output.color.rgb, 1.0 / 2.2), output.color.a);
			return output;
		}
		HLSLEND
	}
	
}