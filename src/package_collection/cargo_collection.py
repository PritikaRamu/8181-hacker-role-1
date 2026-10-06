import requests

url = "https://rubygems.org/names"
output_file = "rubygems_packages.txt"

print("Downloading gem names...")

r = requests.get(url)

names = r.text.splitlines()

with open(output_file, "w") as f:
    for name in names:
        f.write(name + "\n")

print("Total packages:", len(names))