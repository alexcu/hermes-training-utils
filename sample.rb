#!/usr/bin/env ruby

#
# This sample script samples output from tagged Argus data.
#
# Usage:  ./sample.tb /path/to/input/dir \
#                     /path/to/output/dir \
#                     num_times \
#                     [min_bibs] \
#                     [max_bibs]
#
# Where `num_times` is the number of samples you want and `min_bibs`
# and `max_bibs` are the number of minimum/maximum bibs per image desired.
# These are optional.
#

require 'fileutils'
require 'json'

src_dir = ARGV[0]
raise 'Missing input directory argument' if src_dir.nil?
dst_dir = ARGV[1]
raise 'Missing output directory argument' if dst_dir.nil?
num_times = ARGV[2]
raise 'Missing number of times argument' if num_times.nil?
min_bibs = ARGV[3] # min bibs per image
max_bibs = ARGV[4] # max bibs per image
photos = Dir["#{src_dir}/*.jpg"]

num_times = num_times.to_i
min_bibs = min_bibs.to_i unless min_bibs.nil?
max_bibs = max_bibs.to_i unless max_bibs.nil?
check_bibs = min_bibs.is_a?(Integer) || max_bibs.is_a?(Integer)

# Cleanup
FileUtils.rm_r(dst_dir) if Dir.exist?(dst_dir)
FileUtils.mkdir_p(dst_dir)

puts "#{src_dir} -> #{dst_dir} at sample #{num_times}"
sample_space = []
num_times.times.each do
  src_file = photos.sample
  next if sample_space.include?(src_file)
  json_file = "#{src_file}.json"
  if check_bibs
    json_str = File.read(json_file)
    argus_data = JSON.parse(json_str, symbolize_names: true)
    num_people = argus_data[:NumberOfPeopleTagged]
    next unless num_people >= min_bibs || num_people <= max_bibs
  end
  puts "Copy #{src_file} -> #{src_file}[.json]"
  FileUtils.cp(src_file, dst_dir)
  FileUtils.cp(json_file, dst_dir)
  sample_space << src_file
end
