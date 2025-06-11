Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex EquirectangularToCubemapVertex
		#pragma fragment EquirectangularToCubemapFragment

		#include "Core.hlsl"

		struct Attributes
		{
			float3 positionOS : POSITION;
			float2 texcoord : TEXCOORD0;
			uint instanceID : SV_InstanceID;
		};

		struct Varyings
		{
			float4 positionCS : SV_POSITION;
			float2 texcoord : TEXCOORD0;
			uint renderTargetIndex : SV_RenderTargetArrayIndex;
		};

		Varyings EquirectangularToCubemapVertex(Attributes input)
		{
			Varyings output;
			output.positionCS = float4(input.positionOS, 1.0f);
			output.texcoord = input.texcoord;
			output.renderTargetIndex = input.instanceID;
			return output;
		}

		TEXTURE2D(_EquirectangularTexture);	SAMPLER(_EquirectangularTexture_Sampler);

		// https://www.youtube.com/watch?v=wU7Yq_j-tqA
		float3 GetDirection(uint face, float2 uv)
		{
			float3 direction[6] =
			{
				float3(-uv.x, -uv.y, 1.0),
				float3(uv.x, -uv.y, -1.0),
				float3(uv.y, 1.0, uv.x),
				float3(-uv.y, -1.0, uv.x),
				float3(1.0, -uv.y, uv.x),
				float3(-1.0, -uv.y, -uv.x)
			};
			return normalize(direction[face]);
		}

		float3 SampleEquirectangularTexture(float3 direction) 
		{
			float phi = atan2(direction.z, direction.x);
			float theta = asin(direction.y);
			float u = (phi + PI) / (2.0f * PI);
			float v = (theta + PI / 2.0f) / PI;
			return SAMPLE_TEXTURE2D(_EquirectangularTexture, _EquirectangularTexture_Sampler, float2(u, v)).rgb;
		}

		float4 EquirectangularToCubemapFragment(Varyings input) : SV_TARGET
		{
			return float4(SampleEquirectangularTexture(GetDirection(input.renderTargetIndex, input.texcoord * 2.0 - 1.0)), 1);
		}
		HLSLEND
	}
}