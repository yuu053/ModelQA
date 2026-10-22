#define MODELQA_BINDLESS 1
#include "../Mesh.hlsl"

struct MeshPayload { uint meshletIds[32]; };
groupshared MeshPayload sPayload;
groupshared uint sVisibleMask;

bool OutsidePlane(float4 plane, float3 center, float radius)
{
    float distance = dot(plane, float4(center, 1.0f));
    float extent = radius * length(plane.xyz);
    // Invalid arithmetic must retain geometry, never produce false rejection.
    if (!isfinite(distance) || !isfinite(extent)) return false;
    float allowance = 1.0e-5f * (dot(abs(plane), abs(float4(center, 1.0f))) + extent + 1.0f);
    return distance < -extent - allowance;
}

bool SphereVisible(float3 center, float radius)
{
    if (radius < 0.0f || !isfinite(radius)) return true;
    // Row-vector transforms: clip halfspaces are combinations of MVP columns.
    float4 x = float4(gMVP[0][0], gMVP[1][0], gMVP[2][0], gMVP[3][0]);
    float4 y = float4(gMVP[0][1], gMVP[1][1], gMVP[2][1], gMVP[3][1]);
    float4 z = float4(gMVP[0][2], gMVP[1][2], gMVP[2][2], gMVP[3][2]);
    float4 w = float4(gMVP[0][3], gMVP[1][3], gMVP[2][3], gMVP[3][3]);
    return !(OutsidePlane(w + x, center, radius) || OutsidePlane(w - x, center, radius) ||
             OutsidePlane(w + y, center, radius) || OutsidePlane(w - y, center, radius) ||
             OutsidePlane(z, center, radius) || OutsidePlane(w - z, center, radius));
}

[numthreads(32, 1, 1)]
void ASMain(uint lane : SV_GroupIndex, uint3 group : SV_GroupID)
{
    ByteAddressBuffer meshlets = ResourceDescriptorHeap[gMeshletDescriptor];
    if (lane == 0) sVisibleMask = 0;
    GroupMemoryBarrierWithGroupSync();
    uint id = group.x * 32u + lane;
    bool visible = false;
    if (id < meshlets.Load(0)) {
        float4 sphere = asfloat(meshlets.Load4((4u + id * 8u + 4u) * 4u));
        visible = SphereVisible(sphere.xyz, sphere.w);
        if (visible) InterlockedOr(sVisibleMask, 1u << lane);
    }
    GroupMemoryBarrierWithGroupSync();
    // Compact in source order, not the nondeterministic order in which lanes
    // reach an atomic increment. This preserves equal-depth primitive ordering
    // and is independent of implementation wave size (no WaveSize assumption).
    uint visibleMask = sVisibleMask;
    if (visible) {
        uint preceding = (1u << lane) - 1u;
        sPayload.meshletIds[countbits(visibleMask & preceding)] = id;
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(countbits(visibleMask), 1u, 1u, sPayload);
}

[outputtopology("triangle")]
[numthreads(64, 1, 1)]
void MSMain(uint lane : SV_GroupIndex, uint3 group : SV_GroupID,
    in payload MeshPayload data,
    out vertices VSOutput outVertices[64], out indices uint3 triangles[126])
{
    ByteAddressBuffer meshlets = ResourceDescriptorHeap[gMeshletDescriptor];
    ByteAddressBuffer source = ResourceDescriptorHeap[gVertexDescriptor];
    uint id = data.meshletIds[group.x];
    uint4 h = meshlets.Load4((4u + id * 8u) * 4u);
    SetMeshOutputCounts(h.x, h.y);
    if (lane < h.x) {
        uint vertexId = meshlets.Load((h.z + lane) * 4u);
        uint base = vertexId * 72u;
        VSInput v;
        v.position = asfloat(source.Load3(base));
        v.normal = asfloat(source.Load3(base + 12u));
        v.tangent = asfloat(source.Load4(base + 24u));
        v.uv0 = asfloat(source.Load2(base + 40u));
        v.uv1 = asfloat(source.Load2(base + 48u));
        v.color = asfloat(source.Load4(base + 56u));
        outVertices[lane] = VSMain(v);
    }
    for (uint t = lane; t < h.y; t += 64u) {
        uint packed = meshlets.Load((h.w + t) * 4u);
        triangles[t] = uint3(packed & 63u, (packed >> 6u) & 63u, (packed >> 12u) & 63u);
    }
}
