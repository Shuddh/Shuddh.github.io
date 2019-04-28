import csv
import json
from itertools import combinations


def read_data(filename):
    data = {}
    with open(filename, "r", encoding="utf8") as file:
        a = csv.reader(file)
        next(a)
        for row in a:
            tags = row[2][:-1]
            tags = tags.replace("<", "").replace(">", "||")
            if "||" in tags:
                for tag1 in tags.split("||"):
                    for tag2 in tags.split("||"):
                        if tag1 != tag2:
                            if tag1 in data:
                                if tag2 in data[tag1]:
                                    data[tag1][tag2] += 1
                                else:
                                    data[tag1][tag2] = 1
                            else:
                                data[tag1] = {}
                                data[tag1][tag2] = 1
    return data


def sort_data(data):
    sorted_data = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_data[:10]


python_data = read_data("python-tag-raw.csv")
java_data = read_data("java-tag-raw.csv")
js_data = read_data("js-tag-raw.csv")
mysql_data = read_data("mysql-tag-raw.csv")


links = []
nodes = {}
all_tags = set()
py_temp = sort_data(python_data["python"])
java_temp = sort_data(java_data["java"])
js_temp = sort_data(js_data["javascript"])
mysql_temp = sort_data(js_data["mysql"])

all_tags.add("java")
all_tags.add("python")
all_tags.add("javascript")
all_tags.add("mysql")

for k, v in py_temp:
    all_tags.add(k)
for k, v in java_temp:
    if k not in all_tags:
        all_tags.add(k)
for k, v in js_temp:
    if k not in all_tags:
        all_tags.add(k)
for k, v in mysql_temp:
    if k not in all_tags:
        all_tags.add(k)

i = 0
for tag in all_tags:
    nodes[tag.capitalize()] = i
    i+=1
 
temp = []
comb = combinations(all_tags,2)
links.append(("Source","Target","Weight","Label"))
for tag1, tag2 in comb:
    count = 0
    if tag1 in python_data and tag2 in python_data[tag1]:
        count += python_data[tag1][tag2]
    if tag1 in java_data and tag2 in java_data[tag1]:
        count += java_data[tag1][tag2]
    if tag1 in js_data and tag2 in js_data[tag1]:
        count += js_data[tag1][tag2]
    if tag1 in mysql_data and tag2 in mysql_data[tag1] and mysql_data[tag1][tag2] > 5:
        count += mysql_data[tag1][tag2]
    if count > 30 and count <= 100:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 2, count))
    elif count > 100 and count <= 250:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 5, count))
    elif count > 250 and count <= 500:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 8, count))
    elif count > 500 and count <= 1000:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 12, count))
    elif count > 1000 and count <= 5000:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 25, count))            
    elif count > 5000 and count <= 8000:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 40, count))
    elif count > 8000 and count <= 13000:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 60, count))            
    elif count > 1300 and count <= 20000:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 80, count))            
    elif count > 20000:
        links.append((nodes[tag1.capitalize()], nodes[tag2.capitalize()], 100, count))            
  


with open("network-edge.csv", "w", newline="") as file:
    cw = csv.writer(file)
    cw.writerows(links)


node = [("Id", "Label")]
i = 0
for tag in all_tags:
    node.append((i,tag.capitalize()))
    i+=1
with open("network-nodes.csv", "w", newline="") as file:
    cw = csv.writer(file)
    cw.writerows(node)
