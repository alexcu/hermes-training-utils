#!/usr/bin/env ruby

#
# Script to detect people using YOLO mini on Darknet
#
# Usage:
# person_detect.rb /path/to/input/dir \
#                  /path/to/output/dir \
#                  /path/to/darknet/dir
#

require 'json'
require 'open3'

# regions = []
# elapsed_time = -1.0



# STDIN.read.split("\n").each do |line|
#   if line.include?('Predicted in')
#     elapsed_time = /[^:]+: Predicted in (\d+.\d+) seconds./.match(line)[1].to_f
#   elsif line.start_with?('person,')
#     raw_data = line.split(',')
#     regions << {
#       accuracy: raw_data[1].to_f,
#       x1: raw_data[2].to_f,
#       y1: raw_data[3].to_f,
#       x2: raw_data[4].to_f,
#       y2: raw_data[5].to_f
#     }
#   end
# end

# puts JSON.dump(person: { regions: regions, elapsed_time: elapsed_time })

def proc_files(in_dir, out_dir, darknet_dir)
  # Pass this into stdin for darknet (i.e., all files we want to test)
  stdin = Dir["#{in_dir}/*.jpg"].join("\n")
  command = %(
    #{darknet_dir}/darknet detector -i 1 test \
    #{darknet_dir}/cfg/voc.data \
    #{darknet_dir}/cfg/tiny-yolo-voc.cfg \
    #{darknet_dir}/tiny-yolo-voc.weights
  )
  stdout, stderr, status = Open3.capture3(command, stdin_data: stdin)
  puts command, stdout, stderr, status
end

def main
  in_dir = ARGV[0]
  raise 'Input directory missing' if in_dir.nil?

  out_dir = ARGV[1]
  raise 'Output directory missing' if out_dir.nil?

  darknet_dir = ARGV[2]
  raise 'Path to darknet missing' if darknet_dir.nil?

  proc_files(in_dir, out_dir, darknet_dir)
end

main
