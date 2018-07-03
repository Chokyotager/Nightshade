import os
import json
import numpy as np
import random

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

        # Shuffle
        random.shuffle(self.smiles)

        self.smiles_length = len(self.smiles)

        self.smiles_vocabulary = vocabulary
        self.smiles_vocabulary_size = len(self.smiles_vocabulary)

        self.index = 0

        pass

    def getData (self, amount=10, shuffle=True):


        smiles_raw = list()
        smiles = list()
        labels = list()

        if shuffle:
            random.shuffle(self.smiles)

        for i in range(amount):

            smile_raw = self.smiles[self.index % self.smiles_length]
            smile = self.indexSmiles(smile_raw, padding=200)
            label = self.createLabels([self.classes.index(x) for x in self.compounds[smile_raw]["labels"]])

            smiles_raw.append(smile_raw)
            smiles.append(smile)
            labels.append(label)

            self.index += 1

        return smiles, labels, smiles_raw

    def createLabels (self, indices):
        # Create multi-class labels

        labels = np.zeros(shape=[self.classes_amount])

        for index in indices:
            labels += np.eye(self.classes_amount)[index]

        return labels

    def indexSmiles (self, string, padding=0):

        indices = list()

        i = 0

        while i < len(string):
            # Truncation is greedy as per SMILES formatting
            for j in range(3, 0, -1):
                # Check three data points ahead
                truncation = string[i:i+j]
                if truncation in self.smiles_vocabulary:
                    index = self.smiles_vocabulary.index(truncation)
                    indices.append(index + 1)

                    i += j
                    break

                else:
                    continue

            # Ignore if key is unknown.
            i += 1

        if padding > 0:
            pad_amount = padding - len(indices)
            indices = [0] * pad_amount + indices

        return indices

    def labelsToReadable (self, labels):

        indices = list(np.where(labels == 1)[0])
        readable = list()

        for index in indices:
            string_label = self.classes[index]
            readable.append(string_label)

        return readable
