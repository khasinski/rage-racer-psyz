#!/usr/bin/env ruby
# frozen_string_literal: true

source = File.read(ARGV.fetch(0))
block = source[/if \(isTextured && nVertices == 4 &&.*?nIndices \+= 6;\n            \}/m]
abort "textured axis-aligned FT4 has no PS1 right-edge coverage strip" unless block
abort "right-edge strip does not preserve endpoint U accumulator semantics" unless
  block.include?("uStep") && block.include?("edge[0].u = edge[1].u") &&
  block.include?("edge[2].u = edge[3].u")
abort "right-edge strip unexpectedly expands the bottom endpoint" if
  block.match?(/\.y\+\+/)

trace = source.index("TraceGpuPrimitive(vertex_cur, nVertices")
strip = source.index("if (isTextured && nVertices == 4")
abort "primitive trace sees synthetic edge vertices" unless trace && strip && trace < strip

puts "textured FT4 right endpoint is inclusive without stretching original UVs"
