import csv
import pandas as pd

# Read file to get EventId
data = pd.read_csv('../Res/IoTDB1.log_templates.csv')
event_templates = data[['EventId']].drop_duplicates()

# Create increse mapping
event_id_to_mapped = {event_id: idx + 1 for idx, event_id in enumerate(sorted(event_templates['EventId'].unique()))}

# Read sequece file
df = pd.read_csv('../Res/BGL_sequence.csv')

# Chhange sequence to new mapped id
df['sequence'] = df['sequence'].apply(lambda x: [event_id_to_mapped[code] for code in eval(x)])

# Save mapping to use
mapping_df = pd.DataFrame(list(event_id_to_mapped.items()), columns=['EventId', 'MappedId'])
mapping_df.to_csv('../Res/event_id_mapping.csv', index=False)

# Save mapped sequence
df.to_csv('../Res/BGL_sequence_mapped.csv', index=False)
print("Mapping hoàn tất và đã lưu vào file 'BGL_sequence_mapped.csv' và 'event_id_mapping.csv'")