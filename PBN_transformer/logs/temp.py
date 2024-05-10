import os

files = [file for file in os.listdir(".") if file.endswith(".csv")]
print(files)
for file in files:
    with open(file, "r") as file_obj:
        # read first character
        first_char = file_obj.read(1)
        print(f"File: {file}")
        if not first_char:
            os.remove(file)
        else:
            print("File is NOT empty")
    print("-----------------------------------")
