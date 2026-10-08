#!/usr/bin/env ruby

require "set"
require "uri"

ROOT = File.expand_path("..", __dir__)
PICGO_ROOT = File.expand_path("../picgo", ROOT)
PICGO_PREFIX = "https://raw.githubusercontent.com/yy2lyx/picgo/admin/"
URL_PATTERN = %r{#{Regexp.escape(PICGO_PREFIX)}[^\s)'\"<>]+}

def image_urls(pattern)
  Dir.glob(pattern).each_with_object(Set.new) do |path, urls|
    File.read(path, encoding: "UTF-8").scan(URL_PATTERN) { |url| urls << url }
  end
end

source_urls = image_urls(File.join(ROOT, "_posts", "**", "*.{md,markdown}"))
site_urls = image_urls(File.join(ROOT, "_site", "**", "*.html"))

missing_files = source_urls.reject do |url|
  relative_path = URI::DEFAULT_PARSER.unescape(url.delete_prefix(PICGO_PREFIX))
  File.file?(File.join(PICGO_ROOT, relative_path))
end

missing_from_site = source_urls - site_urls
stale_in_site = site_urls - source_urls

issues = {
  "Missing from the local picgo repository" => missing_files,
  "Missing from generated _site pages" => missing_from_site,
  "Stale picgo URLs still present in _site" => stale_in_site
}

issues.each do |label, urls|
  next if urls.empty?

  warn "#{label}:"
  urls.sort.each { |url| warn "  #{url}" }
end

if issues.values.any?(&:any?)
  warn "\nPicgo image validation failed. Rebuild _site after updating article image paths."
  exit 1
end

puts "Validated #{source_urls.length} picgo image URLs across _posts, _site, and #{PICGO_ROOT}."
