#include "MetricSummary.hlsli"
StructuredBuffer<MetricSummary> InputSummaries : register(t0);
RWStructuredBuffer<MetricSummary> OutputSummaries : register(u0);
cbuffer ReductionConstants : register(b0)
{
    uint InputCount;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};
groupshared MetricStatistics SharedStats[256];
groupshared uint SharedHistogram[64];

// Ping-pong distinct buffers; insert a UAV barrier / SRV transition between
// dispatches. Repeat until one summary remains, then read back only 288 bytes.
[numthreads(256,1,1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint inputSize,inputStride,outputSize,outputStride;
    InputSummaries.GetDimensions(inputSize,inputStride);
    OutputSummaries.GetDimensions(outputSize,outputStride);
    MetricStatistics stats = (MetricStatistics)0;
    if (tid.x < InputCount && tid.x < inputSize) stats=InputSummaries[tid.x].statistics;
    SharedStats[lane]=stats;
    if (lane < 64)
    {
        uint total=0;
        const uint begin=group.x*256u;
        const uint end=begin+min(256u,min(InputCount,inputSize)-min(begin,min(InputCount,inputSize)));
        for (uint i=begin;i<end;++i) total+=InputSummaries[i].histogram[lane];
        SharedHistogram[lane]=total;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint stride=128;stride>0;stride>>=1)
    {
        if (lane < stride) SharedStats[lane]=MergeStatistics(SharedStats[lane],SharedStats[lane+stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0 && group.x < outputSize)
    {
        OutputSummaries[group.x].statistics=SharedStats[0];
        for (uint i=0;i<64;++i) OutputSummaries[group.x].histogram[i]=SharedHistogram[i];
    }
}
