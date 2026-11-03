#!/usr/bin/env ruby
# frozen_string_literal: true

source = File.read(ARGV.fetch(0))
flush = source[/void Draw_FlushBuffer\(void\) \{.*?\n\}/m]
abort "cannot find Draw_FlushBuffer" unless flush

abort "textured batches do not synchronize sampling VRAM" unless
  flush.include?("if (batch_has_texture)")
abort "sampling VRAM synchronization still depends on a lossy dirty rectangle" if
  flush.match?(/if \(batch_has_texture && vram_dirty/)
abort "textured batches do not copy complete PS1 VRAM" unless
  flush.match?(/SDL_CopyGPUTextureToTexture\(\s*copy, &vram_src, &vram_dst, VRAM_W, VRAM_H, 1, false\);/m)

puts "textured SDL GPU batches receive a complete frame-current VRAM snapshot"
