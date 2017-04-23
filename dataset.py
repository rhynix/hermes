class Dataset:
    def __init__(self, inputs, outputs):
        assert len(inputs) == len(outputs)

        self.inputs  = inputs
        self.outputs = outputs

    def split_at(self, index):
        return self[:index], self[index:]

    def __getitem__(self, key):
        return Dataset(self.inputs[key], self.outputs[key])

    def __len__(self):
        return len(self.inputs)
