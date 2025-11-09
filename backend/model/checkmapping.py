import pickle

path = r"C:\Users\Admin\PycharmProjects\now\Network Intrusion\backend\model\state_encoder.pkl"
#path = r"C:\Users\Admin\PycharmProjects\now\Network Intrusion\backend\model\protocol_encoder.pkl"

# Load the pickle file
with open(path, "rb") as f:
    encoder = pickle.load(f)

print("Loaded object type:", type(encoder))
from sklearn.preprocessing import LabelEncoder

if isinstance(encoder, LabelEncoder):
    print("Classes:", encoder.classes_)
    # Example: ['normal', 'attack', 'probe', ...]
mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
print("Mapping:", mapping)
