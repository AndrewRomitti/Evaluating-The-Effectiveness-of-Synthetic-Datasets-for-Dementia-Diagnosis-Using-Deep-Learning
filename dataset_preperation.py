import os
import json

dummy_son = {
	"labels":
		[

		]
}

for subdir in sorted(os.listdir(/mydata/Data)):
	for file in os.listdir(/mydata/Data/subdir):
		dummy_son["labels"].append([subdir/file, os.listdir(/mydata/Data/).index(subdir)])

json_object = json.dumps(dummy_son)

with open("/mydata/Data/dataset.json", "w") as outfile:
	outfile.write(json_object)
