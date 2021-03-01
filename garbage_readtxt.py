import os


if __name__ == '__main__':
    root = 'D://DeepLearningData/AdienceBenchmarkGenderAndAgeClassification/'
    txt_pth = os.path.join(root, 'fold_0_data.txt')
    txt = open(txt_pth, 'r')
    lines = txt.readlines()
    lines.pop(0)
    directory, fn, _, age, gender = lines[0].split('\t')[:5]
    print(directory)
    print(fn)
    print(age)
    print(gender)