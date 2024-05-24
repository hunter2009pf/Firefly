import re


def find_max_line_length(file_paths :list[str]) -> int:
    max_length = 0
    # Open the file using with statement to ensure it gets closed after reading
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf8') as file:
            for idx, line in enumerate(file):
                parts = re.split(r'\t+', line)
                if len(parts)>0:
                    current_length = len(parts[0])
                    if current_length > max_length:
                        max_length = current_length
                        print(f"{file_path}: {idx+1}")
    return max_length

if __name__=="__main__":
    file_paths = ["../data/eval.txt", "../data/test.txt", "../data/train.txt"]
    max_length = find_max_line_length(file_paths=file_paths)
    print(f"最大文本长度：{max_length}")