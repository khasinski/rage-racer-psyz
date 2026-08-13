#!/usr/bin/env ruby
# frozen_string_literal: true

source = File.read(ARGV.fetch(0))
common = File.read(ARGV.fetch(1))
%w[PsxTextureTriangleSpan PsxTextureSpanSample ModernTextureSample
   Draw_FillTexturedQuadScanlineGaps].each do |name|
  abort "textured FT4 raster compatibility helper #{name} is missing" unless
    source.include?(name)
end
abort "textured correction does not use the PS1 fixed-point UV integer part" unless
  source.include?("fixed_u >> 16") && source.include?("fixed_v >> 16")
abort "textured correction does not reproduce truncated 16.16 UV steps" unless
  source.include?("step_u") && source.include?("step_v") &&
  source.include?("TextureSamplesDiffer")
abort "textured correction ignores unstable integral float UV boundaries" unless
  source.include?("fabs(raw_u - round(raw_u))") &&
  source.include?("bool unstable = ModernTextureSample")
abort "obsolete axis-aligned geometry strip is still active" if
  source.include?("uStep") || source.include?("Right endpoint:")

trace = source.index("TraceGpuPrimitive(vertex_cur, nVertices")
correction = source.index("Draw_FillTexturedQuadScanlineGaps(compatibility_quad)")
abort "primitive trace sees synthetic compatibility pixels" unless
  trace && correction && trace < correction

puts "textured FT4 scanline endpoints preserve PS1 UVs without stretching geometry"

abort "pixel tracer does not expose exact triangle-edge membership" unless
  common.include?("TraceTriangleEdges") && common.include?('edge=%s%s%s values=%ld,%ld,%ld')
