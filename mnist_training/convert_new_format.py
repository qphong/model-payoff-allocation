"""
The current format of data file in old/ is a dictionary of keys ["party_digits", "name", "test_acc"]
Now, we change to dictionary of keys ["test_acc", "party_labels", "replicated_party_idxs", "name"]
"""
import pickle 
from os import listdir
from os.path import isfile, join

for filename in listdir("old"):
    path = join("old", filename)
    
    if isfile(path) and not path.endswith(".py"):
        print(path)
        
        info = pickle.load(
            open(path, 'rb')
        )
        print(info.keys())

        newinfo = dict(info)
        newinfo["party_labels"] = info["party_digits"]

        del newinfo["party_digits"]

        newinfo["replicated_party_idxs"] = []
        if "replication_1" in filename:
            newinfo["replicated_party_idxs"] = [[1,2]]
        elif "replication_4" in filename:
            newinfo["replicated_party_idxs"] = [[4,5]]

        with open(filename, "wb") as outfile:
            pickle.dump(newinfo, outfile, protocol=pickle.HIGHEST_PROTOCOL)