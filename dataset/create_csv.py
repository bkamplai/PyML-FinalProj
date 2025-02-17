import os
import pandas as pd

"""
Generates a CSV file for the ASL alphabet dataset folder.
Layout (image, labels):
image name, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, blank
ex: 0aff0fc7-568a-40a3-b510-0584d817cd01.rgb_0000.png, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
"""
# Folder where data is stored, name of csv file that will be created
def create_csv(folder, file):
    data = []
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['Blank']

    # Go through each letter in folder
    for letter in os.listdir(folder):
        letter_dir = os.path.join(folder, letter)

        for image in os.listdir(letter_dir):
            row = [image] + [1 if letter == label else 0 for label in labels]
            data.append(row)
    
    # Save to csv file
    df = pd.DataFrame(data, columns=["image name"] + labels)
    df.to_csv(file, index=False)


def main():
    # Directories for train and test folders
    train_folder = 'Train_Alphabet'
    test_folder = 'Test_Alphabet'

    # Create csv files for train and test folders
    create_csv(train_folder, 'train_csv.csv')
    create_csv(test_folder, 'test_csv.csv')


if __name__ == "__main__":
    main()
