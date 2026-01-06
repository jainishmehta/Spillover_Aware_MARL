import pickle
import pprint

# Load the file
with open('results.pkl', 'rb') as f:
    data = pickle.load(f)

# Pretty print the structure
pp = pprint.PrettyPrinter(indent=2, depth=3)
pp.pprint(data)

# If it's a dict, explore keys
if isinstance(data, dict):
    print("\nKeys in pickle file:")
    for key in data.keys():
        print(f"  {key}: {type(data[key])}")