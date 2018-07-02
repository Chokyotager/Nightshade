import os
import json
import numpy as np

data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/tox21/"

data_info = json.loads(open(data_dir + "data.json").read())
vocabulary = [x for x in open(data_dir + "vocabulary.smiles").read().split("\n") if x != ""]

class Data ():

    def __init__ (self):

        # Load data into sequence
        sets = data_info["training"].keys()

        # Map as such:
        # [SMILES, label, drugID]

        data = dict()

        for key in sets:
            current = open(data_dir + data_info["training"][key], "r").read().split("\n")

            for x in current[:-1]:

                split = x.split("\t")

                # If SMILES is unique:
                if split[0] not in data:
                    # Create a new entry
                    data[split[0]] = {"synonym": split[1], "labels": list()}

                # Append label if true
                if int(split[2]) == 1 and key not in data[split[0]]["labels"]:
                    data[split[0]]["labels"].append(key)

        self.classes = list(sets)
        self.classes_amount = len(self.classes)

        self.compounds = data
        self.smiles = list(data.keys())
        self.smiles_length = len(self.smiles)

        self.smiles_vocabulary = vocabulary
        self.smiles_vocabulary_size = len(self.smiles_vocabulary)

        self.index = 0

        pass

    def getData (self, amount=10):

        def createLabels (indices):
            # Create multi-class labels

            labels = np.zeros(shape=[self.classes_amount])

            for index in indices:
                labels += np.eye(self.classes_amount)[index]

            return labels


        smiles_raw = list()
        smiles = list()
        labels = list()

        for i in range(amount):
            smile_raw = self.smiles[self.index % self.smiles_length]
            smile = self.indexSmiles(smile_raw)
            label = createLabels([self.classes.index(x) for x in self.compounds[smile_raw]["labels"]])

            smiles_raw.append(smile_raw)
            smiles.append(smile)
            labels.append(label)

            self.index += 1

        return smiles, labels, smiles_raw

    def indexSmiles (self, string):

        indices = list()

        i = 0

        while i < len(string):
            # Truncation is greedy as per SMILES formatting
            for j in range(3, 0, -1):
                # Check three data points ahead
                truncation = string[i:i+j]
                if truncation in self.smiles_vocabulary:
                    index = self.smiles_vocabulary.index(truncation)
                    indices.append(index)

                    i += j
                    break

                else:
                    continue

            # Ignore if key is unknown.
            i += 1

        return indices
