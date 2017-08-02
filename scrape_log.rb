#!/usr/bin/env ruby

require 'csv'

#
# Script to scrape ML training log
#

log_file = ARGV[0]
raise 'No log file provided!' if log_file.nil?

data = []

def proc_line_for(type, line, data, last_epoch_id)
  if line.start_with?(type)
    key = type.downcase.gsub(/\s/, '_').to_sym
    epoch_data = data.find { |d| d[:epoch_id] == last_epoch_id }
    epoch_data[key] = /(\d+.\d+)/.match(line)[0].to_f
  end
end

last_epoch_id = -1

File.open(log_file, 'r') do |infile|
  while (line = infile.gets)
    if line.start_with?('Epoch')
      epoch_id = %r{Epoch (\d+)\/(\d+)}.match(line)[1].to_i
      data << { epoch_id: epoch_id }
      last_epoch_id = epoch_id
    else
      proc_line_for('Mean number of bounding boxes', line, data, last_epoch_id)
      proc_line_for('Classifier accuracy for bounding boxes', line, data, last_epoch_id)
      proc_line_for('Loss RPN classifier', line, data, last_epoch_id)
      proc_line_for('Loss RPN regression', line, data, last_epoch_id)
      proc_line_for('Loss Detector classifier', line, data, last_epoch_id)
      proc_line_for('Loss Detector regression', line, data, last_epoch_id)
      proc_line_for('Elapsed time', line, data, last_epoch_id)
      proc_line_for('Total loss decreased', line, data, last_epoch_id)
    end
  end
end

# Print headings
CSV do |out|
  out << data[0].keys
end

CSV do |out|
  data.each do |epoch|
    out << epoch.values if epoch.values.length > 1
  end
end
