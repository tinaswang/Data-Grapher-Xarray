from lxml import etree
import json
import os.path
import plotly.graph_objs as go

class Parser(object):

    def __init__(self, filename):

        if os.path.isfile(filename):
            self.filename = filename
        else:
             raise FileNotFoundError(filename)

    def parse(self): # returns parsed data
        data = self.__convert(self.filename, self.__setup(self.filename))
        return data

    def __setup(self, filename): # sets up XML to be usable by etree
        parser = etree.XMLParser(remove_comments=True)
        xmldoc = etree.parse(filename, parser = parser)
        root = xmldoc.getroot()
        return root

    def __convert(self, filename, root): # converts XML to Python dictionary format
        datadictionary = {root.tag:
         {child.tag:
                {grandchild.tag:
                {attribute: self.__getname(attribute, grandchild) for attribute in grandchild.attrib}
            for grandchild in child

                }
             for child in root
            }
        }
        for child in root:
            for grandchild in child:
                if grandchild.text:
                    if "Detector" in grandchild.tag and any("type" in s for s in grandchild.attrib):
                        dimensions = grandchild.attrib["type"].replace("[", " ").replace(",", " ").replace("]", " ").split()
                        if len(dimensions) == 3 and dimensions[0] == "INT32":
                            dimensions.remove("INT32")
                            dimensions =[int(i) for i in dimensions]
                            cleaned_array = self.__arraysplit(grandchild.text.split(), dimensions)
                            datadictionary[root.tag][child.tag][grandchild.tag].update({"data" : cleaned_array})

                    else:
                        try:
                            datadictionary[root.tag][child.tag][grandchild.tag].update({"#text" : float(grandchild.text)})
                        except:
                            datadictionary[root.tag][child.tag][grandchild.tag].update({"#text" : grandchild.text})
        return datadictionary

    def __getname(self, attribute,child): # gets name of attributes
        try:
            return float(child.get(attribute))
        except:
            return child.get(attribute)


    def __arraysplit(self, data, dimensions): # splits large arrays of data
        data = [int(i) for i in data]
        rows = dimensions[0]
        cols = dimensions[1]
        datalist = [data[x:x+cols] for x in range(0, len(data), cols)]
        return datalist


    def dump_as_dict(self): # returns data in Python dictionary form
        return self.parse()

    def dump_as_json(self): # dump data as a JSON string
        data =  self.parse()
        json_str= json.JSONEncoder().encode(data)
        return json_str

    def dump_to_file(self, output_file): # dump JSON to a file
        data=  self.parse()
        with open(output_file, 'w') as f:
            json.dump(data, f)
        print("Dumped to %s" %(output_file))

    def xpath_get(self, path):   # search the resulting Python dictionary

    #@path is the form of "/tag/tag/tag"
        elem = self.dump_as_dict()
        for x in path.strip("/").split("/"):
            elem = elem.get(x)
            if elem is None:
                raise TypeError("No results")
        return elem


def main():
    pass

if __name__ == "__main__":
    main()
