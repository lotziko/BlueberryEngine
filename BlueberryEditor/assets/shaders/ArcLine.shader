Shader
{
	Pass
	{
		Blend One Zero
		ZWrite Off
		Cull None

		HLSLBEGIN
		#pragma vertex ArcLineVertex
		#pragma geometry ArcLineGeometry
		#pragma fragment ArcLineFragment

		#include "Input.hlsl"

		struct Attributes
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: NORMAL;
			float4 fromPositionOSAngle	: TANGENT;
			float4 color				: COLOR;
		};

		struct GeometryVaryings
		{
			float3 centerPositionOS		: POSITION;
			float4 normalOSRadius		: TEXCOORD0;
			float4 tangentOSAngle		: TEXCOORD1;
			float3 bitangentOS			: TEXCOORD2;
			float4 color				: TEXCOORD3;
		};

		struct FragmentVaryings
		{
			float4 positionCS	: SV_POSITION;
			float4 color		: TEXCOORD0;
		};

		GeometryVaryings ArcLineVertex(Attributes input)
		{
			GeometryVaryings output;
			output.centerPositionOS = input.centerPositionOS;
			output.normalOSRadius = input.normalOSRadius;
			output.tangentOSAngle = float4(normalize(input.fromPositionOSAngle.xyz), input.fromPositionOSAngle.w);
			output.bitangentOS = cross(output.normalOSRadius.xyz, output.tangentOSAngle.xyz);
			output.color = input.color;
			return output;
		}

		[maxvertexcount(128)]
		void ArcLineGeometry(point GeometryVaryings input[1], inout LineStream<FragmentVaryings> lineStream)
		{
			float3 center = input[0].centerPositionOS;
			float3 normalOS = input[0].normalOSRadius.xyz;
			float3 tangentOS = input[0].tangentOSAngle.xyz;
			float3 bitangentOS = input[0].bitangentOS;
			float radius = input[0].normalOSRadius.w;
			float angle = input[0].tangentOSAngle.w;
			float4 color = input[0].color;

			float segments = 36.0;
			for (int i = 0; i <= 36; ++i)
			{
				float u = cos(radians(i / segments * angle)) * radius;
				float v = sin(radians(i / segments * angle)) * radius;
				
				FragmentVaryings p1;
				p1.positionCS = mul(mul(float4(center + tangentOS * u + bitangentOS * v, 1.0), _ModelMatrix), _ViewProjectionMatrix);
				p1.color = color;
				lineStream.Append(p1);
			}
			lineStream.RestartStrip();
		}

		float4 ArcLineFragment(FragmentVaryings input) : SV_TARGET
		{
			return input.color;
		}
		HLSLEND
	}
}