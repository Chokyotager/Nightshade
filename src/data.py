import os
import json
import numpy as np
import random
import re

# TODO: Inactive labelling

data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/tox21/"

data_info = json.loads(open(data_dir + "data.json").read())
vocabulary = [x for x in open(data_dir + "vocabulary.smiles").read().split("\n") if x != ""]

class Data ():

    def __init__ (self):

        # Load data into sequence
        sets = sorted(data_info["training"].keys(), key=str.lower)

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
                    data[split[0]] = {"synonym": split[1], "labels": list(), "negative": list()}

                # Append label if true
                if int(split[2]) == 1 and key not in data[split[0]]["labels"]:
                    data[split[0]]["labels"].append(key)

                if int(split[2]) == 0 and key not in data[split[0]]["negative"]:
                    data[split[0]]["negative"].append(key)

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

    def getData (self, amount=10, shuffle=True, map_neutrals=False):

        smiles_raw = list()
        smiles = list()
        labels = list()
        weights = list()

        if shuffle:
            random.shuffle(self.smiles)

        for i in range(amount):

            smile_raw = self.smiles[self.index % self.smiles_length]
            smile = self.indexSmiles(smile_raw, padding=200)
            label, weight = self.createLabels(self.compounds[smile_raw]["labels"], self.compounds[smile_raw]["negative"], map_neutrals=map_neutrals)

            smiles_raw.append(smile_raw)
            smiles.append(smile)
            labels.append(label)
            weights.append(weight)

            self.index += 1

        return smiles, labels, weights, smiles_raw

    def _createProtoLabels (self, labels):

        ret = np.zeros(shape=[self.classes_amount])

        for label in labels:
            ret += np.eye(self.classes_amount)[self.classes.index(label)]

        return ret

    def createLabels (self, positive, negative, map_neutrals=False, neutral_map=0.5):
        # Create multi-class labels

        undocumented = [x for x in self.classes if x not in positive + negative]
        neutrals = self._createProtoLabels(undocumented)

        labels = self._createProtoLabels(positive)

        if map_neutrals:
            labels += neutrals * neutral_map

        weights = np.abs(neutrals - 1)

        return labels, weights

    def createIndexLabels (self, indices):
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

                    i += (j - 1)
                    break

                else:
                    if j == 1:
                        print("Unknown SMILES indexing key {}. Skipping.".format(truncation))
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

class Scorer (Data):

    def __init__ (self):

        # Load data into sequence

        compounds = [x.split("\t") for x in open(data_dir + data_info["testing"]["score"]).read().split("\n")[1:-1]]
        data = dict()

        # Interpret the labels
        scores = open(data_dir + data_info["testing"]["score_sheet"]).read().split("\n")

        keys = sorted(scores[0].split("\t")[1:], key=str.lower)

        labels = [x.split("\t") for x in scores[1:-1]]

        for compound in compounds:
            tag = compound[1]

            # Locate tag in labels
            target = [x for x in labels if x[0] == tag][0][1:]

            positive = [keys[i] for i, x in enumerate(target) if x == "1"]
            negative = [keys[i] for i, x in enumerate(target) if x == "0"]

            data[compound[0]] = {"synonym": tag, "labels": positive, "negative": negative}

        self.classes = keys
        self.classes_amount = len(self.classes)

        self.compounds = data
        self.smiles = list(data.keys())

        self.smiles_length = len(self.smiles)

        self.smiles_vocabulary = vocabulary
        self.smiles_vocabulary_size = len(self.smiles_vocabulary)

        self.index = 0

    def getEvaluations (self):
        return self.getData(amount=self.smiles_length, shuffle=False)

class Validator (Data):

    def __init__ (self):

        labels = open(data_dir + data_info["testing"]["test_sheet"]).read().split("$$$$")[:-1]

        sets = sorted(data_info["training"].keys(), key=str.lower)

        def splitSDF (x):
            # Function splits the individual SDF data
            search = re.findall(">  <(.*?)>\n(.*?)$", x, flags=re.S | re.M)

            positive = list()
            negative = list()

            keys = search[2:]
            for key in keys:
                if key[1] == "1":
                    positive.append(key[0])
                else:
                    negative.append(key[0])

            compound = {"synonym": search[1][1], "labels": positive, "negative": negative}
            return compound

        labels = [splitSDF(x) for x in labels]

        compounds = open(data_dir + data_info["testing"]["test"]).read().split("\n")[:-1]

        data = dict()

        for compound in compounds:
            # Map compounds to dict
            current = compound.split("\t")

            smiles = current[0]
            batch = current[1]

            data[smiles] = [x for x in labels if x["synonym"] == batch][0]

        self.classes = sets
        self.classes_amount = len(self.classes)

        self.compounds = data
        self.smiles = list(data.keys())

        self.smiles_length = len(self.smiles)

        self.smiles_vocabulary = vocabulary
        self.smiles_vocabulary_size = len(self.smiles_vocabulary)

        self.index = 0

    def getValidationSet (self, amount=20):

        smiles, labels, weights, smiles_raw = self.getData(amount=amount, shuffle=False)

        return smiles, labels
