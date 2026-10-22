// ModelQA Procedural Environment Cubemap 生成シェーダー
// 目的:
//   - 将来の本格IBL用に、方向ベース環境をTextureCube相当のTexture2DArrayへ焼く。
//   - 背景、反射、屈折が同じ環境を参照するための入口。
// 注意:
//   - 現行Rendererではまだ未使用でも、shader assetsとして同梱する。
//   - Cubemapを使う場合はDX12側でR16G16B16A16_FLOAT + UAV + mip生成が必要。

#include "ModelQAPreviewEnvironment.hlsli"

RWTexture2DArray<float4> gOutEnvironmentCube : register(u0);

cbuffer EnvCubeGenConstants : register(b0)
{
    uint  gCubeSize;
    uint  gPreset;
    uint  gPadding0;
    uint  gPadding1;

    float4 gLightEnv;   // xyz: light direction, w: environment intensity
    float4 gEnvParams;  // x: light intensity, y: sky yaw, z: cloud amount, w: contrast
};

float3 CubeFaceDirection(uint face, float2 uv)
{
    // uv: -1..1。D3D TextureCubeとして破綻しにくい向きにしている。
    // FXC /Od の古い definite-assignment 解析でも一意になるよう、単一returnにする。
    float3 direction;
    float3 fallback;
    if (face == 0)
    {
        direction = float3( 1.0f, -uv.y, -uv.x); fallback = float3( 1.0f, 0.0f, 0.0f); // +X
    }
    else if (face == 1)
    {
        direction = float3(-1.0f, -uv.y,  uv.x); fallback = float3(-1.0f, 0.0f, 0.0f); // -X
    }
    else if (face == 2)
    {
        direction = float3( uv.x,  1.0f,  uv.y); fallback = float3( 0.0f, 1.0f, 0.0f); // +Y
    }
    else if (face == 3)
    {
        direction = float3( uv.x, -1.0f, -uv.y); fallback = float3( 0.0f,-1.0f, 0.0f); // -Y
    }
    else if (face == 4)
    {
        direction = float3( uv.x, -uv.y,  1.0f); fallback = float3( 0.0f, 0.0f, 1.0f); // +Z
    }
    else
    {
        direction = float3(-uv.x, -uv.y, -1.0f); fallback = float3( 0.0f, 0.0f,-1.0f); // -Z
    }
    return MQA_SafeNormalize(direction, fallback);
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= gCubeSize || tid.y >= gCubeSize || tid.z >= 6)
    {
        return;
    }

    float2 uv = (float2(tid.xy) + 0.5f) / max((float)gCubeSize, 1.0f);
    uv = uv * 2.0f - 1.0f;

    const float3 dir = CubeFaceDirection(tid.z, uv);
    const float3 hdr = MQA_EvaluateModelQAPreviewEnvironment(
        dir,
        (float)gPreset,
        gLightEnv.xyz,
        max(gLightEnv.w, 0.0f),
        max(gEnvParams.x, 0.0f),
        gEnvParams.y,
        gEnvParams.z,
        gEnvParams.w);

    gOutEnvironmentCube[uint3(tid.xy, tid.z)] = float4(hdr, 1.0f);
}
