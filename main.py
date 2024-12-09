import os 

for dirpath, dirnames, filenames in os.walk('./data'):
    print(f'The courant directory is {dirpath}')
    print(f'Found Subcategories: {dirnames}')
    print("-" * 50)
