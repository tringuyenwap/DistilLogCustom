import fasttext.util
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import re
import string

def clean(s):
    """ Preprocess log message
    Parameters
    ----------
    s: str, raw log message
    Returns
    -------
    str, preprocessed log message without number tokens and special characters
    """
    # s = re.sub(r'(\d+\.){3}\d+(:\d+)?', " ", s)
    # s = re.sub(r'(\/.*?\.[\S:]+)', ' ', s)
    s = re.sub('\]|\[|\)|\(|\=|\,|\;', ' ', s)
    s = " ".join([word.lower() if word.isupper() else word for word in s.strip().split()])
    s = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s))
    s = " ".join([word for word in s.split() if not bool(re.search(r'\d', word))])
    trantab = str.maketrans(dict.fromkeys(list(string.punctuation)))
    content = s.translate(trantab)
    s = " ".join([word.lower().strip() for word in content.strip().split()])
    return s

def encoder(s):
    words = s.split(" ")
    words = [word for word in words]
    s = " ".join(words)
    vector = ft.get_sentence_vector(s)
    return vector


ft = fasttext.load_model('../Res/cc.en.300.bin')

# Đọc mapping
mapping_df = pd.read_csv('../Res/event_id_mapping.csv')
event_id_mapping = dict(zip(mapping_df['EventId'], mapping_df['MappedId']))

# Đọc templates và áp dụng mapping
templates_df = pd.read_csv('../Res/IoTDB1.log_templates.csv')
templates_df = templates_df.sort_values('EventId')  # Sắp xếp theo EventId
df = templates_df['EventTemplate'].values

vec = []
for i in range(len(df)):
    clean_template = clean(df[i]).lower()
    vec.append(encoder(clean_template))

estimator30 = PCA(n_components=30)
pca_30 = estimator30.fit_transform(vec)

# PPA: De-averaged
vec30 = []
result = pca_30 - np.mean(pca_30)
pca = PCA(n_components=30)
pca_30 = pca.fit_transform(result)
U = pca.components_
for i, x in enumerate(result):
    for u in U[0:7]:
        x = x - np.dot(u.transpose(), x) * u
    vec30.append(list(x))
vec30 = np.array(vec30)

df_org_vec = pd.DataFrame(vec)
df_org_vec.to_csv("../Res/vector.csv", index = None, header = None)

df_pca_vec = pd.DataFrame(vec30)
df_pca_vec.to_csv("../Res/pca_vector.csv", index = None, header = None)

