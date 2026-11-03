#!/usr/bin/env ruby
# frozen_string_literal: true

shader = File.read(File.expand_path("../src/platform/shaders/psx.frag.glsl", __dir__))
raise "GLSL dither matrix must be indexed column,row as x,y" unless shader.include?("ditherMatrix[dx][dy]")
raise "stale row,column dither indexing" if shader.include?("ditherMatrix[dy][dx]")
raise "integral UV floor boundaries are not stabilized" unless
  shader.include?("fixedHalfUnit") && shader.include?("rawUV + fixedHalfUnit")
puts "GLSL column-major dither matrix is indexed as [x][y]"
