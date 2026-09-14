Shader
{
	Properties
	{
		TextureCube _BaseMap = "white"
	}
	Pass
	{
		Blend One Zero
		ZTest Equal
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex SkyboxVertex
		#pragma fragment SkyboxFragment

		#pragma keyword_global_vertex MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float3 texcoord : TEXCOORD0;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		struct Output
		{
			float4 color : SV_TARGET;
			float depth : SV_DEPTH;
		};

		Varyings SkyboxVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = TransformWorldToClip(CAMERA_POSITION_WS + input.positionOS.xyz);
			output.texcoord = input.positionOS.xyz;

			return output;
		}

		TEXTURECUBE(_BaseMap);		SAMPLER(_BaseMap_Sampler);

		Output SkyboxFragment(Varyings input)
		{
			Output output;
			float4 color = SAMPLE_TEXTURECUBE(_BaseMap, _BaseMap_Sampler, input.texcoord) * pow(2, 2.2); // TODO exposure property
			output.color = ApplyVolumetricFog(color, input.positionCS.xy * CAMERA_SIZE_INV_SIZE.zw, 1);
			output.depth = 1.0;
			return output;
		}
		HLSLEND
	}
	Pass
	{
		Blend One Zero
		ZTest Equal
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex SkyboxVertex
		#pragma fragment SkyboxFragment

		#pragma keyword_global_vertex MULTIVIEW

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			VERTEX_INPUT_INSTANCE_ID
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			VERTEX_OUTPUT_VIEW_INDEX
		};

		struct Output
		{
			float4 color : SV_TARGET;
			float depth : SV_DEPTH;
		};

		Varyings SkyboxVertex(Attributes input)
		{
			Varyings output;
			SETUP_INSTANCE_ID(input);
			SETUP_OUTPUT_VIEW_INDEX(output);

			output.positionCS = float4(input.positionOS, 1.0f);
			return output;
		}

		Output SkyboxFragment(Varyings input)
		{
			Output output;
			float4 color = CAMERA_COLOR;
			output.color = ApplyVolumetricFog(color, input.positionCS.xy * CAMERA_SIZE_INV_SIZE.zw, 1);
			output.depth = 1.0;
			return output;
		}
		HLSLEND
	}
}