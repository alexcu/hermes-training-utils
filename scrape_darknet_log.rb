#!/usr/bin/env ruby

#
# Script to scrape darknet detection log (only want to detect person)
# Usage is to pipe darknet output into stdin. Sample input:
#
#    foo.jpg: Predicted in 2.182727 seconds.
#    person,0.30,632,136,679,227
#    person,0.58,3,127,146,437
#    person,0.48,272,149,432,452
#

require 'json'

regions = []
elapsed_time = -1.0

STDIN.read.split("\n").each do |line|
  if line.include?('Predicted in')
    elapsed_time = /[^:]+: Predicted in (\d+.\d+) seconds./.match(line)[1].to_f
  elsif line.starts_with?('person,')
    raw_data = line.split(',')
    regions << {
      accuracy: raw_data[1],
      x1: raw_data[2],
      y1: raw_data[3],
      x2: raw_data[4],
      y2: raw_data[5]
    }
  end
end

puts JSON.dump(person_regions: regions, elapsed_time: elapsed_time)
