import pandas as pd

# Read file CSV
file_path = '../Res/BGL_sequence_mapped.csv'
df = pd.read_csv(file_path)

# Hàm để chỉnh sửa cột sequence
def format_sequence(sequence):
    try:
        # Chuyển chuỗi thành danh sách các số nguyên
        sequence_list = list(map(int, sequence.strip('[]"').split(',')))
        # Định dạng lại thành chuỗi với khoảng trắng giữa các số
        return ' '.join(map(str, sequence_list))
    except Exception as e:
        print(f"Lỗi khi xử lý sequence: {sequence}, Lỗi: {e}")
        return sequence


df['sequence'] = df['sequence'].apply(format_sequence)

# Lưu lại file CSV đã chỉnh sửa
output_file_path = '../Res/BGL_sequence_mapped.csv'
df.to_csv(output_file_path, index=False)

print(f"Đã chỉnh sửa và lưu file CSV tại: {output_file_path}")
